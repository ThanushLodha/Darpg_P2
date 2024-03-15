from flask import Flask,request,jsonify
import json
import google.generativeai as genai
from transformers import pipeline
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

with open("Files\\parents.txt","r") as file:
    parents = file.readlines()
with open("Files\\parentsCode.txt","r") as file:
    parentsCode = file.readlines()
with open("Files\\Class1_Child.json","r") as file:
    Class1_Children = json.load(file)
with open("Files\\Class2_Child.json","r") as file:
    Class2_Children = json.load(file)
with open("Files\\Class3_Child.json","r") as file:
    Class3_Children = json.load(file)
with open("Files\\Class4_Child.json","r") as file:
    Class4_Children = json.load(file)
with open("Files\\Class5_Child.json","r") as file:
    Class5_Children = json.load(file)
with open("Files\\Class6_Child.json","r") as file:
    Class6_Children = json.load(file)
with open("Files\\Class7_Child.json","r") as file:
    Class7_Children = json.load(file)
with open("Files\\Class8_Child.json","r") as file:
    Class8_Children = json.load(file)

for i in range (0,len(parentsCode)):
    parentsCode[i] = parentsCode[i].strip("\n")
    parentsCode[i] = int(parentsCode[i])
for i in range(len(parents)):
  parents[i] = parents[i].strip("\n")

classifier = pipeline("zero-shot-classification")
@app.route('/classify', methods=['POST'])
def classify_grievance():
    # print("Classifying")
    if request.method == 'POST':
        # print("In method")
        data = request.json
        # print(data)
        grievance = data.get('grievance')

        if grievance:
            output_list = []
            summary = model.generate_content(f"Summarize the given in English, do not add any other things which aren't mentioned in the text: {grievance}")
            parent = model.generate_content(f"Given the grievance organisations of India: {parents}\n\n\nTo which organisation does this belong to: {summary.text},\n\n Don't give any additional organisations which aren't mentioned, just print the label as it is")
            parent = parent.text
            output_list = [parent]
            count = 0
            for i in range(1, 9):
                child_list = []
                for child in globals()[f"Class{i}_Children"]:
                    if child['parent'] == parent:
                        child_list = child['child']
                print(count)
                if len(child_list) != 0:
                    count = count + 1
                    possible_labels = child_list
                    result = classifier(summary.text, possible_labels)
                    output_list.append(str(result['labels'][0]))
                    with open("hig.txt", "w") as f:
                        f.write(str(result['labels']))
                        f.write("\n")
                        f.write(str(result['scores']))
                        f.write("\n")
                        f.write("================================")
                    parent = result['labels'][0]
                    child_list = []
                else:
                    break
            final_output = ""
            for i in output_list:
                final_output = final_output + i + " >> "
            final_output = final_output[:-3]
            # print(final_output)
            return jsonify({'output_list': final_output})
        else:
            return jsonify({'error': 'No grievance provided'})

if __name__ == '__main__':
    app.run(port=8000)