import sys
import json

sys.path.append('/data/christinefang/unified-model-editing')

from glue_eval.sst_eval import SSTEval
from glue_eval.mrpc_eval import MRPCEval
from glue_eval.cola_eval import COLAEval
from glue_eval.rte_eval import RTEEval
from glue_eval.mmlu_eval import MMLUEval
from glue_eval.sentiment_analysis_eval import SENTIMENT_ANALYSIS_Eval
from glue_eval.dialogue_eval import DIALOGUE_Eval
from glue_eval.nli_eval import NLIEval
from datasets import load_dataset
import random

class GLUEEval():
    def __init__(self, model, tokenizer, number_of_tests = None, sst_number_of_few_shots = 0, mrpc_number_of_few_shots = 0, cola_number_of_few_shots = 0, rte_number_of_few_shots = 0, mmlu_number_of_few_shots = 0, sentiment_analysis_number_of_few_shots = 0, nli_number_of_few_shots = 0, dialogue_number_of_few_shots = 0):
        self.model = model

        self.tokenizer = tokenizer

        self.sst_eval = SSTEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sst_number_of_few_shots)

        self.mrpc_eval = MRPCEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mrpc_number_of_few_shots)

        self.cola_eval = COLAEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = cola_number_of_few_shots)

        self.rte_eval = RTEEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = rte_number_of_few_shots)

        self.mmlu_eval = MMLUEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = mmlu_number_of_few_shots)

        self.sentiment_analysis_eval = SENTIMENT_ANALYSIS_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = sentiment_analysis_number_of_few_shots)

        self.nli_eval = NLIEval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = nli_number_of_few_shots)

        self.dialogue_eval = DIALOGUE_Eval(model, tokenizer, number_of_tests = number_of_tests, number_of_few_shots = dialogue_number_of_few_shots)


    def _save_generations(self, record_path, generations, task):
        #store individual generation file
        output_filename = record_path.replace('.json', '_' + task + '_gen.json')
        with open(output_filename, "w") as f:
            json.dump(generations, f, indent=4)



    def evaluate(self, glue_results, sst_flag = False, mmlu_flag = False, mrpc_flag = False, cola_flag = False, rte_flag = False, nli_flag = False, sentiment_analysis_flag = False, dialogue_flag = False, gen_len = 5):
        if sst_flag:
            result_dict, generations = self.sst_eval.evaluate(gen_len)
            glue_results['sst'] = {
                'score': result_dict,
                'generations': generations
            }

        if mmlu_flag:
            result_dict, generations = self.mmlu_eval.evaluate(gen_len)
            glue_results['mmmlu'] = {
                'score': result_dict,
                'generations': generations
            }

        if mrpc_flag:
            result_dict, generations = self.mrpc_eval.evaluate(gen_len)
            glue_results['mrpc'] = {
                'score': result_dict,
                'generations': generations
            }

        if cola_flag:
            result_dict, generations = self.cola_eval.evaluate(gen_len)
            glue_results['cola'] = {
                'score': result_dict,
                'generations': generations
            }

        if rte_flag:
            result_dict, generations = self.rte_eval.evaluate(gen_len)
            glue_results['rte'] = {
                'score': result_dict,
                'generations': generations
            }

        if sentiment_analysis_flag:
            result_dict, generations = self.sentiment_analysis_eval.evaluate(gen_len)
            glue_results['sentiment_analysis'] = {
                'score': result_dict,
                'generations': generations
            }

        if nli_flag:
            result_dict, generations = self.nli_eval.evaluate(gen_len)
            glue_results['nli'] = {
                'score': result_dict,
                'generations': generations
            }
            
        if dialogue_flag:
            result_dict, generations = self.dialogue_eval.evaluate(gen_len)
            glue_results['dialogue'] = {
                'score': result_dict,
                'generations': generations
            }
        return glue_results


        

