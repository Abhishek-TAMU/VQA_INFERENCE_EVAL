import json
import matplotlib.pyplot as plt
from vqa import VQA
from vqa_eval import VQAEval

# Set up file names and paths
dataDir = 'datasets'
versionType = 'v2_'         # Use '' when using VQA v2.0 dataset if needed
taskType    = 'OpenEnded'   # 'OpenEnded' for v2.0; for v1.0 can be 'OpenEnded' or 'MultipleChoice'
dataType    = 'mscoco'      
dataSubType = 'val2014'
annFile     = f'{dataDir}/Annotations/{versionType}{dataType}_{dataSubType}_annotations.json'
quesFile    = f'{dataDir}/Questions/{versionType}{taskType}_{dataType}_{dataSubType}_questions.json'
imgDir      = f'{dataDir}/Images/{dataType}/{dataSubType}/'

# Use the saved output JSON file from inference as the results file.
# (This file is expected to contain an array of result objects with "question_id" and "answer".)
resultType  = "vqav2"
model_type  = 'llama_vision'
output_json_path = f'{dataDir}/Results/{model_type}/{versionType}{taskType}_{dataType}_{dataSubType}_{resultType}_results.json'

# Generate file names for the evaluation output files.
fileTypes   = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']
[accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
    f'{dataDir}/Results/{model_type}/{versionType}{taskType}_{dataType}_{dataSubType}_{resultType}_{fileType}.json'
    for fileType in fileTypes
]

# Create VQA object using the full annotations and question files.
vqa = VQA(annFile, quesFile)

# This change lets you evaluate only on the questions present in your output file.
vqaRes = vqa.loadRes(output_json_path, quesFile)

# Create VQAEval object (n is the precision of accuracy, default is 2)
vqaEval = VQAEval(vqa, vqaRes, n=2)

# Optionally, if you wish to evaluate only the questions present in the results file,
# you can retrieve the list of question IDs from the results.
predicted_ids = [res['question_id'] for res in json.load(open(output_json_path))]
vqaEval.evaluate(quesIds=predicted_ids)

# Print accuracies.
print("\nOverall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
print("Per Question Type Accuracy:")
for quesType, acc in vqaEval.accuracy['perQuestionType'].items():
    print("%s : %.02f" % (quesType, acc))
print("\nPer Answer Type Accuracy:")
for ansType, acc in vqaEval.accuracy['perAnswerType'].items():
    print("%s : %.02f" % (ansType, acc))
print("\n")

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

# Save evaluation results to the Results folder.
with open(accuracyFile, 'w') as f:
    json.dump(vqaEval.accuracy, f)
with open(evalQAFile, 'w') as f:
    json.dump(vqaEval.evalQA, f)
with open(evalQuesTypeFile, 'w') as f:
    json.dump(vqaEval.evalQuesType, f)
with open(evalAnsTypeFile, 'w') as f:
    json.dump(vqaEval.evalAnsType, f)

print('Results saved to:')
print('Accuracy file: %s' % (accuracyFile))
print('EvalQA file: %s' % (evalQAFile))
print('EvalQuesType file: %s' % (evalQuesTypeFile))
print('EvalAnsType file: %s' % (evalAnsTypeFile))
