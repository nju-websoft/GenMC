# coding=utf-8
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import trange, tqdm
import numpy as np
import os
import json
import random
from torch import nn
import torch
import re



@torch.no_grad()
def eval(model, test_examples, tokenizer, eval_batch_size, path_save_result=None, best_dev_acc=None):
    count, count_right = 0, 0
    results = []
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        input_ids, input_masks, output_ids, answer_labels, options_list = get_input_feature(batch_example, max_len, max_len_gen)
        output_sequences = model(input_ids, input_masks, output_ids, do_train=False)

        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        for p, a in zip(predicts, answer_labels):
            if p == a:
                count_right += 1
            count += 1
        save_result(batch_example, predicts)
        results += batch_example
    if path_save_result is not None and best_dev_acc is None:
        save_dataset(path_save_result, results)
    elif path_save_result is not None and best_dev_acc < count_right / count:
        save_dataset(path_save_result, results)
    return count_right / count

def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        for sample in dataset:
            sample = json.dumps(sample, indent=2)
            f.write(sample + '\n')


def save_result(samples, generation):
    label_map = {'1': 'A', '2': 'B', '3': 'C', '4': "D", '5': 'E', '6':'F', '7':'G', '8':'H'}
    for i, sample in enumerate(samples):
        sample['predict_explain'] = generation[i]
        if generation[i] in label_map:
            sample['predict_option'] = label_map[generation[i]]
        else:
            sample['predict_option'] = 'A'
        # for o_i, (opt, opt_name) in enumerate(zip(sample['question']['choices'], 'ABCDE')):

def clean(content: str):
    content = content.lower()
    content = re.sub(r"\(.*?\)", "\1", content)
    return content

def get_input_feature(samples, max_source_length, max_len_gen):
    input_text, output_answer = [], []
    options_list, answer_labels = [], []

    for sample in samples:
        answerKey = sample['answerKey']
        question = clean(sample['question']['stem'])
        input_text_full = task
        options = []
        for o_i, (opt, opt_name) in enumerate(zip(sample['question']['choices'], 'ABCDEFGH')):
            opt['text'] = clean(opt['text'])
            input_text_full += " choice" + str(o_i+1) + ": " + opt['text']
            options.append(opt['text'])
            if answerKey == opt_name:
                output_answer.append(str(o_i+1))
                answer_labels.append(str(o_i+1))
        input_text_full += " question: " + question
        input_text.append(input_text_full)
        options_list.append(options)

    input_encoding = tokenizer(input_text,
                                 padding='longest',
                                 max_length=max_source_length,
                                 truncation=True,
                                 return_tensors="pt")
    input_ids = input_encoding.input_ids.to(device)
    input_masks = input_encoding.attention_mask.to(device)

    output_encoding = tokenizer(output_answer,
                              padding='longest',
                              max_length=max_len_gen,
                              truncation=True,
                              return_tensors="pt")
    output_ids = output_encoding.input_ids.to(device)

    output_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
        output_ids
    ]
    output_ids = torch.tensor(output_ids, dtype=torch.long).to(device)
    return input_ids, input_masks, output_ids, answer_labels, options_list


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


device = torch.device("cuda:0")
class MyT5ForConditionalGeneration(nn.Module):

    def __init__(self, model_path):
        super(MyT5ForConditionalGeneration, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)

    def forward(self, input_ids, input_masks, output_ids, do_train=False):
        if do_train:
            t5_output = self.t5_model(input_ids=input_ids, attention_mask=input_masks, labels=output_ids)
            loss_ans = t5_output.loss
            return loss_ans
        else:
            t5_output = self.t5_model.generate(
                input_ids,
                attention_mask=input_masks,
                do_sample=False,  # disable sampling to test if batching affects output
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            output_sequences = t5_output.sequences
            return output_sequences


def read_dataset(path):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # if i>30:
            #     break
            line = line.strip()
            line = json.loads(line)
            dataset.append(line)
    return dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    max_len = 512
    max_len_gen = 100
    epoch_num = 30
    lrs = [1e-5]
    model_names = ['t5-large']
    tasks = ['qasc']
    # tasks = ['arc-challenge', 'arc-easy']
    # tasks = ['qasc']
    seeds = [0]
    ignore_flag = True
    for lr in lrs:
        for model_name in model_names:
            for task in tasks:
                for seed in seeds:
                    init_model_path = None
                    # init_model_path = './outputs/' + task + '/' + model_name + '/0/pytorch_model.bin'
                    choice_num = None
                    # model_path = '/data1/PTLM/' + model_name + '/'
                    model_path = model_name
                    gradient_accumulation_steps = 1
                    if model_name == '3b' or model_name == 'large' or model_name == 'unifiedqa_3b' or model_name == '11b':
                        train_batch_size = 8
                        test_batch_size = 6
                        num_attention_heads = 16
                    elif model_name == "base":
                        train_batch_size = 8
                        test_batch_size = 8
                        num_attention_heads = 12
                    else:
                        num_attention_heads = 8
                        train_batch_size = 8
                        test_batch_size = 8

                    if task == 'obqa':
                        choice_num = 4
                        data_path_train = './data/obqa/dev.jsonl'
                        data_path_dev = './data/obqa/test.jsonl'
                        data_path_test = './data/obqa/test.jsonl'
                    elif task == 'csqa':
                        choice_num = 5
                        data_path_train = './data/csqa/train.jsonl'
                        data_path_dev = './data/csqa/dev.jsonl'
                        data_path_test = './data/csqa/test.jsonl'
                    elif task == 'piqa':
                        gradient_accumulation_steps = 4
                        choice_num = 2
                        data_path_train = './data/piqa/train.jsonl'
                        data_path_dev = './data/piqa/dev.jsonl'
                        data_path_test = './data/piqa/test.jsonl'
                    elif task == 'arc-easy':
                        choice_num = 4
                        data_path_train = './data/arc-easy/train.jsonl'
                        data_path_dev = './data/arc-easy/dev.jsonl'
                        data_path_test = './data/arc-easy/test.jsonl'
                        # gradient_accumulation_steps = 4
                    elif task == 'arc-challenge':
                        choice_num = 4
                        data_path_train = './data/arc-challenge/train.jsonl'
                        data_path_dev = './data/arc-challenge/dev.jsonl'
                        data_path_test = './data/arc-challenge/test.jsonl'
                    elif task == 'qasc':
                        choice_num = 8
                        data_path_train = './data/qasc/train.jsonl'
                        data_path_dev = './data/qasc/dev.jsonl'
                        data_path_test = './data/qasc/test.jsonl'
                        gradient_accumulation_steps = 1

                    output_model_path = './outputs/' + task + '/' + model_name + '/'
                    path_save_result = './results/' + task + '/' + model_name + '/lr_' + str(lr) + '/seed_' + str(
                        seed) + '/'
                    os.makedirs(path_save_result, exist_ok=True)

                    set_seed(seed)
                    train_examples = read_dataset(data_path_train)
                    dev_examples = read_dataset(data_path_dev)
                    test_examples = read_dataset(data_path_test)
                    print('train_examples:', len(train_examples))
                    print('dev_examples:', len(dev_examples))
                    print('test_examples:', len(test_examples))
                    train_batch_size = train_batch_size // gradient_accumulation_steps
                    tokenizer = T5Tokenizer.from_pretrained(model_path)
                    model = MyT5ForConditionalGeneration(model_path)

                    if init_model_path is not None:
                        checkpoint = torch.load(init_model_path, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])

                    n_gpu = torch.cuda.device_count()
                    if n_gpu == 1:
                        model.cuda()
                    #     model = torch.nn.DataParallel(model)

                    order = list(range(len(train_examples)))
                    random.shuffle(order)

                    best_dev_acc, best_test_acc = 0, 0
                    step_count = 0
                    rouge_score, rouge_score_count = 0, 1
                    early_stop = 0
                    step_all = 0


                    best_dev_rouge_score = 0
                    # eval(model, dev_examples, tokenizer, test_batch_size)
                    count_right, count = 0, 0
                    count_right_t, count_t = 0, 0
                    tr_loss, nb_tr_steps = 0, 0

                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                    from transformers import get_linear_schedule_with_warmup

                    step_count = len(train_examples) // train_batch_size
                    if step_count * train_batch_size < len(train_examples):
                        step_count += 1
                    t_total = epoch_num * step_count
                    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                    #                                             num_warmup_steps=int(0.1 * t_total),
                    #                                             num_training_steps=t_total)
                    # best_dev_acc = eval(model, dev_examples, tokenizer, test_batch_size)
                    # best_test_acc = eval(model, test_examples, tokenizer, test_batch_size)

                    for epoch in range(epoch_num):
                        early_stop += 1
                        model.train()
                        step_count = len(train_examples) // train_batch_size
                        if step_count * train_batch_size < len(train_examples):
                            step_count += 1
                        step_trange = trange(step_count)
                        for step in step_trange:
                            step_count += 1
                            step_all += 1
                            beg_index = step * train_batch_size
                            end_index = min((step + 1) * train_batch_size, len(train_examples))
                            order_index = order[beg_index:end_index]
                            batch_example = [train_examples[index] for index in order_index]
                            input_ids, input_masks, output_ids, answer_labels, options_list = get_input_feature(
                                batch_example, max_len, max_len_gen)
                            loss = model(input_ids, input_masks, output_ids, do_train=True)
                            # if n_gpu > 1:
                            loss = loss.mean()
                            tr_loss += loss.item()
                            nb_tr_steps += 1
                            loss = loss / gradient_accumulation_steps
                            loss.backward()
                            if (step + 1) % gradient_accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                            # scheduler.step()

                            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps, 4))
                            step_trange.set_postfix_str(loss_show)

                        dev_acc = eval(model, dev_examples, tokenizer, test_batch_size,
                                       path_save_result + '/dev.jsonl', best_dev_acc)
                        if dev_acc > best_dev_acc:
                            early_stop = 0
                            best_dev_acc = dev_acc
                            output_model_file = output_model_path + '/'
                            os.makedirs(output_model_file, exist_ok=True)
                            output_model_file += 'pytorch_model.bin'
                            # torch.save({
                            #     'model_state_dict': model.state_dict(),
                            #     'optimizer_state_dict': optimizer.state_dict()
                            # }, output_model_file)
                            print('new best dev acc')
                            test_acc = eval(model, test_examples, tokenizer, test_batch_size,
                                            path_save_result=path_save_result + '/test.jsonl')
                            best_test_acc = test_acc
                            print('new best acc:', dev_acc, test_acc)

                        print('Epoch:', epoch)
                        if early_stop >= 5:
                            break
                    print('seed:', seed, 'lr:', lr, 'model_name:', model_name, 'task:', task)
                    print('best dev acc:', best_dev_acc, 'best_test_acc:', best_test_acc)


