# coding=utf-8
from transformers import T5Tokenizer
from tqdm import trange
import os
import random
import torch
from utils import compute_rouges, save_dataset, read_dataset, set_seed, save_model
from model.modeling_genmc import GenMC
import json
import argparse

device = torch.device("cuda:0")


def get_input_feature(samples, max_source_length, max_len_gen, choice_num, external_sent_num=None):
    sep = ' \\n '
    output_clue = []
    answers = []
    input_ids_q, attention_mask_q = [], []
    input_ids_qo, attention_mask_qo = [], []
    for sample in samples:
        if 'answerKey' in sample:
            answerKey = sample['answerKey']
        else:
            answerKey = "A"
        question = sample['question']['stem']
        while len(sample['question']['choices']) < choice_num:
            sample['question']['choices'].append({"text": "error", "para": "", "label":chr(ord('A')+len(sample)-1)})
        for o_i, (opt, opt_name) in enumerate(zip(sample['question']['choices'], 'ABCDEFGH'[:choice_num])):
            option = opt['text']
            content = ""
            if external_sent_num is not None and 'para' in opt:
                para = opt["para"]
                if isinstance(para, list):
                    if len(para) > external_sent_num:
                        para = para[:external_sent_num]
                    content = sep + " ".join(para)
                elif isinstance(para, str):
                    para = para.split(".")
                    if len(para) > external_sent_num:
                        para = para[:external_sent_num]
                    content = sep + " ".join(para)
                else:
                    print('lack retrieval')
                    # exit(0)
            input_ids_qo.append(question + sep + option + content)


        input_ids_q.append(question + sep)
        if answerKey in '123456':
            answer = ord(answerKey) - ord('1')
        else:
            answer = ord(answerKey) - ord('A')
        answers.append(answer)
        output_clue.append(sample['question']['choices'][answer]['text'])

    def tokenizer_fun(input_ids, max_len):
        encoding = tokenizer(input_ids,
                             padding='longest',
                             max_length=max_len,
                             truncation=True,
                             return_tensors="pt")
        ids = encoding.input_ids.to(device)
        mask = encoding.attention_mask.to(device)
        return ids, mask

    q_ids, q_mask = tokenizer_fun(input_ids_q, max_source_length)
    qo_ids, qo_mask = tokenizer_fun(input_ids_qo, max_source_length)
    clue_ids, _ = tokenizer_fun(output_clue, max_len_gen)
    clue_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
        clue_ids
    ]
    clue_ids = torch.tensor(clue_ids, dtype=torch.long).to(device)
    answers = torch.tensor(answers, dtype=torch.long).to(device)
    return q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue


@torch.no_grad()
def eval(model, test_examples, tokenizer, eval_batch_size, choice_num, max_len, max_len_gen, external_sent_num):
    count, count_right = 0, 0
    results = []
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    sources, targets = [], []
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue = get_input_feature(batch_example,
                                                                                           max_len, max_len_gen,
                                                                                           args.choice_num,
                                                                                           external_sent_num)
        scores, output_sequences = model(q_ids, q_mask, qo_ids, qo_mask, choice_num)

        scores = scores.cpu().detach().tolist()
        answers = answers.cpu().detach().tolist()
        p_anss = []
        for p, a, example in zip(scores, answers, batch_example):
            p_ans = p.index(max(p))
            p_anss.append(example['question']['choices'][p_ans]['label'])
            if p_ans == a:
                count_right += 1
            count += 1
        for sample, p_ans in zip(batch_example, p_anss):
            qid = sample['id']
            results.append(qid + "," + p_ans)
        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        sources += predicts
        targets += output_clue

    rouge_score = compute_rouges(sources, targets)['rouge-l']

    return count_right / count, rouge_score, results



dataset = 'arc_challenge'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default='t5-base',
                        type=str)
    parser.add_argument("--choice_num",
                        default=4,
                        type=int)
    parser.add_argument("--data_path_train",
                        default=f'./data/{dataset}/in_house/train.jsonl',
                        type=str)
    parser.add_argument("--data_path_dev",
                        default=f'./data/{dataset}/in_house/dev.jsonl',
                        type=str)
    parser.add_argument("--data_path_test",
                        default=f'./data/{dataset}/in_house/test.jsonl',
                        type=str)
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_len",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_len_gen",
                        default=64,
                        type=int,
                        help="The maximum total output sequence length for decoder")
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--num_hidden_layers',
                        type=int,
                        default=1,
                        help="The number of hidden layer for co-matching and encoder-decoder interaction transformer")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.5)
    parser.add_argument('--beta',
                        type=float,
                        default=1)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument("--name_save_prix",
                        default=dataset,
                        type=str)
    parser.add_argument("--gpu",
                        default="0",
                        type=str)
    parser.add_argument('--external_sent_num',
                        type=int,
                        default=None,
                        help="The number of retrieved sentences")
    args = parser.parse_args()
    file_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}_ga_{args.gradient_accumulation_steps}_layer_num_{args.num_hidden_layers}_alpha_{args.alpha}_beta_{args.beta}'
    output_model_path = './outputs/' + args.name_save_prix + '/' + file_name + "/"
    path_save_result = './results/' + args.name_save_prix + '/' + file_name + "/"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)
    train_examples = read_dataset(args.data_path_train)
    dev_examples = read_dataset(args.data_path_dev)
    test_examples = read_dataset(args.data_path_test)

    print(json.dumps({"lr": args.lr, "model": args.model_path, "seed": args.seed,
                      "bs": args.train_batch_size,
                      'gradient_accumulation_steps': args.gradient_accumulation_steps,
                      "epoch": args.epoch_num,
                      "train_path": args.data_path_train,
                      "dev_path": args.data_path_dev,
                      "test_path": args.data_path_test,
                      "train_size": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'num_hidden_layers': args.num_hidden_layers,
                      'external_sent_num': args.external_sent_num,
                      "alpha": args.alpha, "beta": args.beta}, indent=2))

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = GenMC(args.model_path, args.num_hidden_layers, args.alpha, args.beta)

    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    tr_loss, nb_tr_steps = 0, 0

    best_dev_acc, _, _ = eval(model, dev_examples, tokenizer, args.eval_batch_size, args.choice_num, args.max_len,
                              args.max_len_gen, args.external_sent_num)
    print('best_dev_acc:',best_dev_acc)
    best_test_acc = 0
    for epoch in range(args.epoch_num):
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        model.train()
        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            q_ids, q_mask, qo_ids, qo_mask, clue_ids, answers, output_clue = get_input_feature(
                batch_example,
                max_source_length=args.max_len,
                max_len_gen=args.max_len_gen,
                choice_num=args.choice_num,
                external_sent_num=args.external_sent_num)
            loss = model(q_ids, q_mask, qo_ids, qo_mask, args.choice_num, clue_ids, answers)

            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps, 4))
            step_trange.set_postfix_str(loss_show)

        dev_acc, dev_rouge_score, results_dev = eval(model, dev_examples, tokenizer, args.eval_batch_size,
                                                     args.choice_num, args.max_len, args.max_len_gen,
                                                     args.external_sent_num)
        print('dev_acc:', dev_acc)
        if dev_acc > best_dev_acc:
            save_dataset(path_save_result + '/dev.csv', results_dev)
            early_stop = 0
            test_acc, test_rouge_score, results_test = eval(model, test_examples, tokenizer, args.eval_batch_size,
                                                            args.choice_num, args.max_len, args.max_len_gen,
                                                            args.external_sent_num)
            save_dataset(path_save_result + '/test.csv', results_test)
            best_dev_acc, best_test_acc, best_dev_rouge_score, best_test_rouge_score = dev_acc, test_acc, dev_rouge_score, test_rouge_score

            # save_model(output_model_path, model, optimizer)
            print('new best dev acc:', dev_acc, 'test_acc:', test_acc, 'rouge:', dev_rouge_score)

        if early_stop >= 10:
            break

    print('best dev acc:', best_dev_acc, 'best_test_acc:', best_test_acc,
          'best_dev_rouge_score:', best_dev_rouge_score, 'best_test_rouge_score:', best_test_rouge_score)
