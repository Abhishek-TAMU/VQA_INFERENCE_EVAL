__author__ = 'aagrawal'
__version__ = '0.9'

import json
import datetime
import copy
from datetime import timezone

class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :param question_file (str): location of VQA question file
        :return: None
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if annotation_file is not None and question_file is not None:
            print('loading VQA annotations and questions into memory...')
            time_t = datetime.datetime.now(timezone.utc)
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            with open(question_file, 'r') as f:
                questions = json.load(f)
            print(datetime.datetime.now(datetime.timezone.utc) - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToQA = {ann['image_id']: [] for ann in self.dataset['annotations']}
        qa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        qqa = {ann['question_id']: [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToQA[ann['image_id']].append(ann)
            qa[ann['question_id']] = ann
        for ques in self.questions['questions']:
            qqa[ques['question_id']] = ques
        print('index created!')

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return: None
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s' % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. Defaults skip filters if not provided.
        :param imgIds: list of image IDs to filter questions
        :param quesTypes: list of question types to filter
        :param ansTypes: list of answer types to filter
        :return: list of question IDs
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if len(imgIds) != 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['question_id'] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. Defaults skip filters if not provided.
        :param quesIds: list of question IDs to filter images
        :param quesTypes: list of question types to filter
        :param ansTypes: list of answer types to filter
        :return: list of image IDs
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset['annotations']
        else:
            if len(quesIds) != 0:
                anns = [self.qa[quesId] for quesId in quesIds if quesId in self.qa]
            else:
                anns = self.dataset['annotations']
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
        ids = [ann['image_id'] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids: integer or list of question IDs
        :return: list of qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns: list of annotation objects to display
        :return: None
        """
        if len(anns) == 0:
            return
        for ann in anns:
            quesId = ann['question_id']
            print("Question: %s" % (self.qqa[quesId]['question']))
            for ans in ann['answers']:
                print("Answer %d: %s" % (ans['answer_id'], ans['answer']))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param resFile: file name of result file
        :param quesFile: file name of question file
        :return: result VQA object
        """
        res = VQA()
        with open(quesFile, 'r') as f:
            res.questions = json.load(f)
        res.dataset['info'] = copy.deepcopy(self.questions.get('info', {}))
        res.dataset['task_type'] = copy.deepcopy(self.questions.get('task_type', ''))
        res.dataset['data_type'] = copy.deepcopy(self.questions.get('data_type', ''))
        res.dataset['data_subtype'] = copy.deepcopy(self.questions.get('data_subtype', ''))
        res.dataset['license'] = copy.deepcopy(self.questions.get('license', ''))

        print('Loading and preparing results...')
        time_t = datetime.datetime.now(datetime.timezone.utc)
        with open(resFile, 'r') as f:
            anns = json.load(f)
        # print("TEST", len(anns["annotations"]), anns["annotations"][0])
        # anns = anns['annotations'] if 'annotations' in anns else anns
        assert type(anns) == list, 'Results is not an array of objects'
        annsQuesIds = [ann['question_id'] for ann in anns]
        assert set(annsQuesIds) == set(self.getQuesIds()), (
            'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids '
            'in annotation file or there is at least one question id that does not belong to the question ids in the annotation file.'
        )
        for ann in anns:
            quesId = ann['question_id']
            if res.dataset.get('task_type', '') == 'Multiple Choice':
                assert ann['answer'] in self.qqa[quesId]['multiple_choices'], (
                    'Predicted answer is not one of the multiple choices'
                )
            qaAnn = self.qa[quesId]
            ann['image_id'] = qaAnn['image_id']
            ann['question_type'] = qaAnn['question_type']
            ann['answer_type'] = qaAnn['answer_type']
        print('DONE (t=%0.2fs)' % ((datetime.datetime.now(datetime.timezone.utc) - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
