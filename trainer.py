from transformers import AdamW
from torch.nn import CrossEntropyLoss, BCELoss
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score,roc_curve,precision_recall_curve,auc,matthews_corrcoef
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm, trange
from funcsigs import signature
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from funcsigs import signature
from torch.nn import CrossEntropyLoss, BCELoss
from torch_geometric.loader import DataLoader
import pickle

import logging
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def save(model, optimizer,output_path):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_path)

def get_bert_optimizer(args,model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer

def train_mimic(args, train_dataset, model, test_dataset):
  tb_writer = SummaryWriter()
  # num_train_epochs = 1
  train_dataloader=DataLoader(train_dataset.get_all_items(), batch_size = args.train_batch_size, shuffle = True)
  if args.max_steps > 0:
      t_total = args.max_steps
      args.num_train_epochs = args.max_steps // (
          len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
      t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  optimizer = get_bert_optimizer(args, model)

  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataloader))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d",
            args.train_batch_size)
  logger.info("  Gradient Accumulation steps = %d",
            args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  all_eval_results = []
  best_results=0
  best_metric=None
  model.zero_grad()
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
  set_seed(args)
  for epoch_i in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc='Iteration')
    for step, batch in tqdm(enumerate(train_dataloader)):
      model.train()
      if args.pure_bert:
        note_ids,full_input_ids_batch, full_segment_ids_batch, full_input_mask_batch, labels = batch
        logit = model(note_ids.to(args.device),full_input_ids_batch.to(args.device), full_segment_ids_batch.to(args.device), full_input_mask_batch.to(args.device))
      else:
        note_ids,full_input_ids_batch, full_segment_ids_batch, full_input_mask_batch, all_graphs, labels = batch
        logit = model(note_ids.to(args.device),full_input_ids_batch.to(args.device), full_segment_ids_batch.to(args.device), full_input_mask_batch.to(args.device), all_graphs)
      loss_fct = BCELoss()
      loss = loss_fct(logit, labels.type(torch.float).to(args.device))

      if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

      loss.backward()
      torch.nn.utils.clip_grad_norm_(
          model.parameters(), args.max_grad_norm)

      tr_loss += loss.item()

      if (step + 1) % args.gradient_accumulation_steps == 0:
        # scheduler.step()  # Update learning rate schedule
        optimizer.step()
        model.zero_grad()
        global_step += 1

        if args.logging_steps > 0 and global_step % args.logging_steps == 0:
            results, eval_loss = evaluate(args, test_dataset, model,epoch_i,"checkpoint_{}".format(global_step))
            if sum(results.values())>best_results and results['rp80']!=0:
                output_folder = args.output_dir+'/model/'+"checkpoint_{}".format(global_step)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_path=output_folder+'/model.pth'
                save(model, optimizer,output_path)
                args.tokenizer.save_pretrained(output_folder)

                logger.info("Saving model checkpoint to %s", output_path)
                logger.info("Saving optimizer and scheduler states to %s", output_path)

                best_results=sum(results.values())
                best_metric=results
            all_eval_results.append(results)
            add_result={}
            add_result['eval_loss']=eval_loss
            add_result['train_loss']=loss.item()
            write_result(args,add_result)
            for key, value in results.items():
                tb_writer.add_scalar(
                    'eval_{}'.format(key), value, global_step)
            tb_writer.add_scalar('eval_loss', eval_loss, global_step)
            tb_writer.add_scalar(
                'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
            logging_loss = tr_loss
                
      if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    results, eval_loss = evaluate(args, test_dataset, model,epoch_i,"checkpoint_{}".format(global_step))
    if sum(results.values())>best_results and results['rp80']!=0:
      output_folder = args.output_dir+'/model/'+"checkpoint_{}".format(global_step)
      if not os.path.exists(output_folder):
          os.makedirs(output_folder)
      output_path=output_folder+'/model.pth'
      save(model, optimizer,output_path)
      args.tokenizer.save_pretrained(output_folder)

      logger.info("Saving model checkpoint to %s", output_path)
      logger.info("Saving optimizer and scheduler states to %s", output_path)

      best_results=sum(results.values())
      best_metric=results
    all_eval_results.append(results)
    add_result={}
    add_result['eval_loss']=eval_loss
    add_result['train_loss']=loss.item()
    write_result(args,add_result)
    for key, value in results.items():
        tb_writer.add_scalar(
            'eval_{}'.format(key), value, global_step)
    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
    tb_writer.add_scalar(
        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
    logging_loss = tr_loss
    if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break

  tb_writer.close()
  return global_step, tr_loss/global_step, all_eval_results,best_metric


def evaluate(args, eval_dataset, model,epoch=0,checkpoint=''):
  results = {}
  eval_dataloader = DataLoader(eval_dataset.get_all_items(), batch_size= args.eval_batch_size, shuffle = False)

  # Eval
  logger.info("***** Running evaluation *****")
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)

  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  model.eval()
  for batch in tqdm(eval_dataloader):
    with torch.no_grad():
      if args.pure_bert:
        note_ids,full_input_ids_batch, full_segment_ids_batch, full_input_mask_batch, labels = batch
        logits = model(note_ids.to(args.device),full_input_ids_batch.to(args.device), full_segment_ids_batch.to(args.device), full_input_mask_batch.to(args.device))
      else:
        note_ids,full_input_ids_batch, full_segment_ids_batch, full_input_mask_batch, all_graphs, labels = batch
        logits = model(note_ids.to(args.device),full_input_ids_batch.to(args.device), full_segment_ids_batch.to(args.device), full_input_mask_batch.to(args.device), all_graphs)
      loss_fct = BCELoss()
      tmp_eval_loss = loss_fct(logits, labels.type(torch.float).to(args.device))
      labels=labels.detach().cpu().numpy()
      tmp_eval_loss=tmp_eval_loss.detach().cpu()
      logits=logits.detach().cpu().numpy()

      eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    
    if preds is None:
        preds=logits
        out_label_ids = labels
    else:
        preds = np.append(preds, logits, axis=0)
        out_label_ids = np.append(
            out_label_ids, labels, axis=0)

  eval_loss = eval_loss / nb_eval_steps
  pred_label=[1 if i>=0.5 else 0 for i in preds]
  result = compute_metrics(pred_label, out_label_ids,preds,epoch, args)
  results.update(result)

  output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
  with open(output_eval_file, 'a+') as writer:
    logger.info('***** Eval results *****')
    logger.info("  eval loss: %s", str(eval_loss))
    writer.write(checkpoint)
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')
    writer.write('\n')

  return result, eval_loss

def write_result(args,result):
  output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
  with open(output_eval_file, 'a+') as writer:
    # logger.info('***** Eval results *****')
    # logger.info("  eval loss: %s", str(eval_loss))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("  %s = %s\n" % (key, str(result[key])))
        writer.write('\n')
    writer.write('\n')
        

def compute_metrics(preds, labels,pred_logits,epoch, args):
  acc,f1=acc_and_f1(preds, labels)
  prediction_df=pd.DataFrame({'pred_logits':pred_logits,'pred':preds,'label':labels})
  prediction_df.to_csv(args.output_dir+str(epoch)+"/prediction.csv", sep=',', encoding='utf-8', index=False, header=True)
  # df_test = pd.read_csv(args.dataset_folder+'test.csv')
  df_test = pd.read_csv(args.dataset_folder+'train.csv')
  fpr, tpr, df_out,auc_score = vote_score(df_test, preds, args.output_dir+'/'+str(epoch)+'/')
  string = '/logits_clinicalbert_readmissions.csv'
  df_out.to_csv(args.output_dir+str(epoch)+string)
  rp80,area = vote_pr_curve(df_test, preds, args.output_dir+'/'+str(epoch)+'/')
  
  out=acc_and_f1(preds, labels)
  out['rp80']=rp80
  out['AUROC']=auc_score
  out['AUPRC']=area
  
  return out

def simple_accuracy(preds, labels):
  return (preds == labels).mean()

def acc_and_f1(preds, labels):
  acc = simple_accuracy(preds, labels)
  f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
  return {
      "acc": acc,
      "f1": f1
  }

def vote_score(df, score,out):
  df['pred_score'] = score
  df_sort = df.sort_values(by=['ID'])
  #score 
  temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
  x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
  df_out = pd.DataFrame({'logits': temp.values, 'ID': x})

  fpr, tpr, thresholds = roc_curve(x, temp.values)
  auc_score = auc(fpr, tpr)

  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  # plt.show()
  string = 'auroc_clinicalbert.png'
  plt.savefig(out+string)
  # plt.savefig(os.path.join(out, string))

  return fpr, tpr, df_out,auc_score

def pr_curve_plot(y, y_score,out):
  precision, recall, _ = precision_recall_curve(y, y_score)
  area = auc(recall,precision)
  step_kwargs = ({'step': 'post'}
                 if 'step' in signature(plt.fill_between).parameters
                 else {})
  
  plt.figure(2)
  plt.step(recall, precision, color='b', alpha=0.2,
           where='post')
  plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall curve: AUC={0:0.3f}'.format(
            area))
  
  string = 'auprc_clinicalbert.png'
  plt.savefig(out+string)

  # plt.savefig(os.path.join(out, string))
  return area


def vote_pr_curve(df, score, out):
  df['pred_score'] = score
  df_sort = df.sort_values(by=['ID'])
  #score 
  temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
  y = df_sort.groupby(['ID'])['Label'].agg(np.min).values
  
  precision, recall, thres = precision_recall_curve(y, temp)
  pr_thres = pd.DataFrame(data =  list(zip(precision, recall, thres)), columns = ['prec','recall','thres'])
  vote_df = pd.DataFrame(data =  list(zip(temp, y)), columns = ['score','label'])
  
  area=pr_curve_plot(y, temp, out)
  
  temp = pr_thres[pr_thres.prec > 0.799999].reset_index()
  
  rp80 = 0
  if temp.size == 0:
      print('Test Sample too small or RP80=0')
  else:
      rp80 = temp.iloc[0].recall
      print('Recall at Precision of 80 is {}', rp80)

  return rp80,area
