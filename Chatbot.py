import spacy
import json
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_lg") # used the _lg spacy model cause it has the most word vectors or something like that
intents = json.loads(open("intents.json").read())

# uses the similarity function to calculate how similar two spacy doc objects are (uses word vectors somehow, not exactly sure how tho)
def compare(topic, doc):
    TmpTopic = nlp(topic)
    TmpDoc = nlp(doc)
    sim = TmpDoc.similarity(TmpTopic)
    return sim

# turns the patterns into a single string and then lists them so they can be converted to spacy doc objects
def list_patterns(intents):
    pattern_list = []
    for intent in intents["intents"]:
        p = ''
        for pattern in intent["patterns"]:
            p = p + ' ' + pattern
        pattern_list.append(p)
    return pattern_list

# creates a dictionary where the patterns are the key, and the tag is the value so that I can get the correct topic name easily for printing at the end
def make_dict(intents):
    topic_dict = {}
    pattern_list = list_patterns(intents)
    count = 0
    for intent in intents["intents"]:
        topic_dict[pattern_list[count]] = intent["tag"]
        count += 1
    return topic_dict

# most of the program, the function works through the topic_combination list and compares every item in the list to the document using the compare function
# and then returns the topic/doc combo with the most similarity
def most_similar(patterns_list, doc, similarity, topic, NameDict):
    if patterns_list == []:
        if topic == 0:
            return "noanswer"
        else:
            return NameDict[topic]
    else:
        sim = compare(patterns_list[0], doc)
        if sim > similarity:
            return most_similar(patterns_list[1:], doc, sim, patterns_list[0], NameDict)
        else:
            return most_similar(patterns_list[1:], doc, similarity, topic, NameDict)

# just a main function to call all the functions as desired
def main():
    print("Say something to begin")
    while True:
        incoming = input()
        topic = most_similar(list_patterns(intents), incoming, 0, 0, make_dict(intents))
        for intent in intents["intents"]:
            if intent["tag"] == topic:
                print(intent["responses"][0])

# runs main
if __name__ == "__main__":
    main()
