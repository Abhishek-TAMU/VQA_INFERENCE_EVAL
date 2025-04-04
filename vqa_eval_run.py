import sys
import os
import json
import random
import matplotlib.pyplot as plt
import skimage.io as io

# Set up file names and paths
# dataDir = '../../VQA'
dataDir = 'datasets'
# sys.path.insert(0, f'{dataDir}/PythonHelperTools/vqaTools')
from vqa import VQA
from vqa_eval import VQAEval

versionType = 'v2_'  # this should be '' when using VQA v2.0 dataset
taskType    = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType    = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'val2014'
annFile     = f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    = f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir      = f'{dataDir}/Images/{dataType}/{dataSubType}/'
resultType  = "vqav2"
model_type  = 'llama_vision'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

# Generate file names for the various output files.
[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
    f'{dataDir}/Results/{model_type}/{versionType}{taskType}_{dataType}_{dataSubType}_{resultType}_{fileType}.json'
    for fileType in fileTypes
]

# Create VQA object and load results.
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# Create VQAEval object (n is the precision of accuracy, default is 2)
vqaEval = VQAEval(vqa, vqaRes, n=2)

# Evaluate results.
# If you have a list of question ids on which you would like to evaluate your results,
# pass it as a list to the evaluate() function. By default, it uses all the question ids.
vqaEval.evaluate()

# Print accuracies.
print("\n")
print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
print("Per Question Type Accuracy is the following:")
for quesType, acc in vqaEval.accuracy['perQuestionType'].items():
    print("%s : %.02f" % (quesType, acc))
print("\n")
print("Per Answer Type Accuracy is the following:")
for ansType, acc in vqaEval.accuracy['perAnswerType'].items():
    print("%s : %.02f" % (ansType, acc))
print("\n")

# Demonstrate how to use evalQA to retrieve low score results.
evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId] < 35]  # 35 is per question percentage accuracy
# if len(evals) > 0:
#     print('Ground truth answers:')
#     randomEval = random.choice(evals)
#     randomAnn = vqa.loadQA(randomEval)
#     vqa.showQA(randomAnn)

#     print('\n')
#     print('Generated answer (accuracy %.02f)' % (vqaEval.evalQA[randomEval]))
#     ann = vqaRes.loadQA(randomEval)[0]
#     print("Answer:   %s\n" % (ann['answer']))

#     imgId = randomAnn[0]['image_id']
#     imgFilename = 'COCO_' + dataSubType + '_' + str(imgId).zfill(12) + '.jpg'
#     if os.path.isfile(os.path.join(imgDir, imgFilename)):
#         I = io.imread(os.path.join(imgDir, imgFilename))
#         plt.imshow(I)
#         plt.axis('off')
#         plt.show()

# Plot accuracy for various question types.
plt.bar(range(len(vqaEval.accuracy['perQuestionType'])),
        list(vqaEval.accuracy['perQuestionType'].values()),
        align='center')
plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])),
           list(vqaEval.accuracy['perQuestionType'].keys()),
           rotation="vertical", fontsize=10)
plt.title('Per Question Type Accuracy', fontsize=10)
plt.xlabel('Question Types', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.show()

# Save evaluation results to the ./Results folder.
with open(accuracyFile, 'w') as f:
    json.dump(vqaEval.accuracy, f)
with open(evalQAFile, 'w') as f:
    json.dump(vqaEval.evalQA, f)
with open(evalQuesTypeFile, 'w') as f:
    json.dump(vqaEval.evalQuesType, f)
with open(evalAnsTypeFile, 'w') as f:
    json.dump(vqaEval.evalAnsType, f)
print('Results saved to the following files:')
print('Accuracy file: %s' % (accuracyFile))
print('EvalQA file: %s' % (evalQAFile))
print('EvalQuesType file: %s' % (evalQuesTypeFile))
print('EvalAnsType file: %s' % (evalAnsTypeFile))
