import numpy as np
import os
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves", "having", "makes","great", "greater", "soon", "sounds", "hard", "help", "email",
    "big", "know","iwould","looking", "using", "morethan", "come", "outside", "particular",
    "writes","write","article"])

def data_select():
    ROOT = "../../RawData/news/20NewsGroup-CountData"
    # get selected classes' labels
    subfields_filename = os.path.join(ROOT, "subfields.txt")
    subfields = open(subfields_filename, 'r').read().split("\n")[:-1]
    subfields = [int(s.split("\t")[2]) for s in subfields]
    # get all instances' labels
    label_filename = os.path.join(ROOT, "train.label")
    labels = open(label_filename, "r").read().split("\n")[:-1]
    labels = [int(s) for s in labels]
    print("total instances number:" + str(len(labels)))
    # get selected instances' index, index start from 1
    category_index = {}
    for i in subfields:
        category_index[i] = []
    for idx, i in enumerate(labels):
        if i in subfields:
            category_index[i].append(idx)
    sum = 0
    for i in subfields:
        sum += len(category_index[i])
    print("selected instances number:" + str(sum))
    # get all instances' term and count
    term_filename = os.path.join(ROOT, "train.data")
    term = open(term_filename, "r").read().split("\n")[:-1]
    term = [s.split(" ") for s in term]
    term = [[int(i) for i in s] for s in term]
    print(term[:10])
    term_map = {}
    term_count = {}
    for i in term:
        try:
            term_map[i[0]].append(i[1])
            term_count[i[0]].append(i[2])
        except:
            term_map[i[0]] = [i[1]]
            term_count[i[0]] = [i[2]]
    # save term_map.txt and term_count.txt
    term_map_filename = os.path.join(ROOT, "term_map.txt")
    term_count_filename = os.path.join(ROOT, "term_count.txt")
    term_map_file = open(term_map_filename, "w")
    term_count_file = open(term_count_filename, "w")
    for i in subfields:
        for idx in category_index[i]:
            term_map_str = str(idx) + " " + str(i) + " "
            term_count_str = str(idx) + " " + str(i) + " "
            for j in range(len(term_map[idx])):
                term_map_str += str(term_map[idx][j]) + " "
                term_count_str += str(term_count[idx][j]) + " "
            term_map_file.writelines(term_map_str[:-1] + "\n")
            term_count_file.writelines(term_count_str[:-1] + "\n")
    term_map_file.close()
    term_count_file.close()


class Text2Image(object):
    def __init__(self, name):
        self.name = name

    def get_info(self):
        self.root = os.path.join("../../RawData/", self.name)
        content_root = os.path.join(self.root, "content")
        instances_number = 2007
        origin_data = []
        data = []
        for i in range(instances_number):
            filename = os.path.join(content_root, str(i + 1) + ".txt")
            s = open(filename, "r", encoding="UTF-8").read()
            origin_data.append(s)
            data.append(s)
        self.origin_data = origin_data

        return
        #triple title
        for i in range(instances_number):
            title = " " + data[i].split("\n")[0]
            for j in range(3):
                data[i] += title

        rule = ["--([\s\S]*)","__([\s\S]*)",".*@.*","[^a-zA-Z ]"]
        for r in rule:
            data = [ re.sub(r,"",s) for s in data]
        data = [ re.sub("\ ([\ ]+)"," ",s) for s in data]
        for i in range(instances_number):
            filename = os.path.join(self.root, "new-content",str(i+1)+".txt")
            open(filename,"w",encoding="UTF-8").writelines(data[i])

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        stop_words=STOP_WORDS)
        self.tf = tf_vectorizer.fit_transform(data)
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           stop_words=STOP_WORDS)
        self.tfidf = tfidf_vectorizer.fit_transform(data)
        self.feature = tfidf_vectorizer.get_feature_names()


        print("tf result shape: %s" % (str(self.tf.shape)))
        print("tf-idf result shape: %s" % (str(self.tfidf.shape)))

    def save_tfidf_keyword(self):
        if not hasattr(self, "tfidf"):
            self.get_info()
        file = open( os.path.join(self.root, "tfidf_keyword.txt"),"w")
        tfidf = self.tfidf.toarray()
        for i in range(self.tfidf.shape[0]):
            idx = tfidf[i,:].argsort()[::-1]
            idx = idx[:20]
            s = ""
            for j in idx:
                s += self.feature[j] + " "
            s += "\n"
            file.writelines(s)
        file.close()

    def save_tf_keyword(self):
        if not hasattr(self, "tf"):
            self.get_info()
        file = open( os.path.join(self.root, "tf_keyword.txt"),"w")
        tf = self.tf.toarray()
        for i in range(self.tfidf.shape[0]):
            idx = tf[i,:].argsort()[::-1]
            idx = idx[:20]
            s = ""
            for j in idx:
                s += self.feature[j] + " "
            s += "\n"
            file.writelines(s)
        file.close()

    def save_topic_model_keyword(self):
        if not hasattr(self,"tf"):
            self.get_info()
        lda = LatentDirichletAllocation(n_components=4, max_iter=5,
                                        learning_method="batch",
                                        random_state=0)
        topics = lda.fit_transform(self.tf)
        file = open( os.path.join(self.root, "LDA_keyword.txt"),"w")
        tf = self.tfidf.toarray()
        for i in range(self.tf.shape[0]):
            mask = tf[i,:].reshape(1,-1).repeat(repeats=4, axis = 0)
            masked_term = lda.components_ * mask
            idx = np.dot( topics[i,:].reshape(1,-1), masked_term ).reshape(-1).argsort()[::-1]
            idx = idx[:20]
            s = ""
            for j in idx:
                s += self.feature[j] + " "
            s += "\n"
            file.writelines(s)
        file.close()

    def wordcloud_generator(self,filename):
        file = open(os.path.join(self.root, filename),"r")
        text_all = file.read().split("\n")[:-1]
        out_pathname = os.path.join( self.root, "wordcloud",  os.path.splitext(filename)[0])
        if not os.path.exists(out_pathname):
            os.mkdir(out_pathname)
        for i in range(len(text_all)):
            wordcloud = WordCloud(background_color="white", prefer_horizontal=1.0).generate_from_text(text_all[i])
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(os.path.join(out_pathname, str(i+1)+".jpg"))
            plt.close()

    def first_sentence_keyword(self):
        data = self.origin_data
        file = open(os.path.join(self.root, "first_sentence_keyword.txt"), "w")
        en = ["zero", "one", "two", "three","four", "five", "six", "seven", "eight", "nine"]
        nu = [0,1,2,3,4,5,6,7,8,9]
        for i in range(len(data)):
            split_data = re.sub(".*@.*","",data[i] )
            split_data = split_data.split("\n")[1:]
            split_data = [ s + " " for s in split_data]
            split_data = "".join(split_data)
            # for i in nu:
            #     split_data = re.sub(str(i),en[i],split_data)
            # split_data = ". " + split_data
            index = re.sub("[\.|\?]\s+","++-" ,split_data)
            index = index.split("++-")
            # print(index)
            # exit()
            while( len(index)>0 and len(index[0]) < 2):
                index = index[1:]
            try:
                index = re.sub(">"," ",index[0])
                index = re.sub("\|"," ",index)
                index = re.sub("-"," ",index)
                index = re.sub("\*"," ",index)
                index = re.sub("\s\s+"," ",index)
                file.writelines(index + "\n")
            except:
                file.writelines("\n")
            # print(s)
            # exit(0)
            # for j in range(2):
        file.close()

    def title_keyword(self):
        data = self.origin_data
        file = open(os.path.join(self.root, "title_keyword.txt"), "w")
        for i in range(len(data)):
            title = data[i].split("\n")[0]
            title = re.sub("Re: ","",title)
            title = re.sub("RE: ","",title)
            title = re.sub("re: ","",title)
            title = re.sub("\*","",title)
            try:
                file.writelines(title + "\n")
            except:
                file.writelines("\n")

if __name__ == "__main__":
    # data_select()
    T = Text2Image("news")
    T.get_info()
    # T.save_tf_keyword()
    # T.save_tfidf_keyword()
    # T.save_topic_model_keyword()
    T.first_sentence_keyword()
    # T.title_keyword()
    # T.wordcloud_generator("LDA_keyword_simple.txt")
