# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from pythia.tasks.vqa.vizwiz import VizWizDataset
from pythia.utils.text_utils import word_tokenize

from pythia.utils.objects_to_byte_tensor import enc_obj2bytes

class TextVQADataset(VizWizDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(dataset_type, imdb_file_index, config, *args, **kwargs)
        self._name = "textvqa"

    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                    pred_source.append('OCR')
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append('VOCAB')
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "answer": pred_answer,
                "pred_source": pred_source,
            }

            predictions.append(entry)

        return predictions

    def add_answer_info(self, sample_info, sample):
        sample_has_answer = ("answers" in sample_info)
        if sample_has_answer:
            # Load real answers from sample_info
            answers = sample_info["answers"]
            sample.gt_answers_enc = enc_obj2bytes(answers)
            answer_processor_arg = {
                "answers": answers,
                "context_tokens": sample.context_tokens,
            }
            processed_answers = self.answer_processor(answer_processor_arg)

            assert not self.config.fast_read, \
                'In M4CTextVQADataset, online OCR sampling is incompatible ' \
                'with fast_read, so fast_read is currently not supported.'
            sample.targets = processed_answers["answers_scores"]
            sample.sampled_idx_seq = processed_answers["sampled_idx_seq"]
            sample.train_prev_inds = processed_answers["train_prev_inds"]
            sample.train_loss_mask = processed_answers["train_loss_mask"]
        else:
            # Load dummy answers as placeholders
            answer_params = self.config.processors.answer_processor.params
            sample.sampled_idx_seq = None
            sample.train_prev_inds = torch.zeros(
                answer_params.max_copy_steps, dtype=torch.long
            )

        return sample
