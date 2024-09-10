import numpy as np

from BayesNetworkClass import BayesNet
from BayesNodeClass import BayesNode
from utilityFuncs import enumeration_ask, clean_str
import urllib.request
import tarfile

if __name__ == '__main__':
    #2.1
    TaksideveiSpec = ("Taksidevei", [], 0.05)
    ApatiSpec = ("Apati", ["Taksidevei"], {True: 0.01, False: 0.004})
    AgoraEksoterikouSpec = ("AgoraEksoterikou", ["Apati" , "Taksidevei"], {(True, True): 0.90, (True, False): 0.1, (False, True): 0.90, (False, False): 0.01})
    DiatheteiIpologistiSpec = ("DiatheteiIpologisti", [], 0.6)
    AgoraDiadiktiouSpec = ("AgoraDiadiktiou", ["DiatheteiIpologisti" , "Apati"], {(True, True): 0.02, (True, False): 0.01, (False, True): 0.011, (False, False): 0.002})
    AgoraSxetikiMeIpologistiSpec = ("AgoraSxetikiMeIpologisti", ["DiatheteiIpologisti"], {True: 0.1, False: 0.001})

    bn = BayesNet()
    bn.add(TaksideveiSpec)
    bn.add(ApatiSpec)
    bn.add(AgoraEksoterikouSpec)
    bn.add(DiatheteiIpologistiSpec)
    bn.add(AgoraDiadiktiouSpec)
    bn.add(AgoraSxetikiMeIpologistiSpec)
    #2.2
    print(enumeration_ask('Apati', {} , bn).show_approx())
    print(enumeration_ask('Apati', dict(AgoraEksoterikou=True, AgoraDiadiktiou =False, AgoraSxetikiMeIpologisti=True), bn).show_approx())
    print(enumeration_ask('Apati', dict(AgoraEksoterikou=True, AgoraDiadiktiou =False, AgoraSxetikiMeIpologisti=True, Taksidevei=True), bn).show_approx())


    #3
    urllib.request.urlretrieve("http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz",
                               "enron1.tar.gz")

    emails = []
    y = []
    with tarfile.open("enron1.tar.gz", "r:gz") as tar:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
                if 'enron1/ham' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    y.append('ham')
                elif 'enron1/spam' in member.name:
                    emails.append(content.decode('utf-8', errors='ignore'))
                    y.append('spam')

    emails = [clean_str(email) for email in emails]

    np.random.seed(13)

    N = len(emails)
    idx = np.random.permutation(N)

    emails_train = [emails[idx[i]] for i in range(int(0.8 * N))]
    y_train = [y[idx[i]] for i in range(int(0.8 * N))]

    emails_test = [emails[idx[i]] for i in range(int(0.8 * N), N)]
    y_test = [y[idx[i]] for i in range(int(0.8 * N), N)]

    #3.1
    ham_counter = 0
    spam_counter = 0
    for status in y_train:
        if status == "ham":
            ham_counter += 1
        else:
            spam_counter += 1

    Pcham = ham_counter / len(y_train)
    Pcspam = spam_counter / len(y_train)
    y_pred = []
    for email_check in emails_test:
        Pwcham = []
        Pwcspam = []
        Pdcham = 1
        Pdcspam = 1
        for word in email_check:
           ham_counter_w = 0
           spam_counter_w = 0

           for i in range(0, len(y_train)): #Find number of times a word exists in each category of emails
               if y_train[i] == "ham" and (word in emails_train[i]):
                   ham_counter_w += 1
               elif y_train[i] == "spam" and (word in emails_train[i]):
                   spam_counter_w += 1
           if ham_counter_w != 0 or spam_counter_w != 0:
               Pwcham.append(ham_counter_w / ham_counter)
               Pwcspam.append(spam_counter_w / spam_counter)
        for x in Pwcham:
            Pdcham = Pdcham * x
        for x in Pwcspam:
            Pdcspam = Pdcspam * x

        if (Pdcham * Pcham) > (Pdcspam * Pcspam):
            y_pred.append("ham")
        else:
            y_pred.append("spam")

    correct = 0
    for i in range(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            correct += 1

    print("Accuracy: ", correct / len(y_test))

    #3.3
    y_pred = []
    for email_check in emails_test:
        Pwcham = []
        Pwcspam = []
        Pdcham = 1
        Pdcspam = 1
        for word in email_check:
            ham_counter_w = 0
            spam_counter_w = 0

            for i in range(0, len(y_train)):  # Find number of times a word exists in each category of emails
                if y_train[i] == "ham" and (word in emails_train[i]):
                    ham_counter_w += 1
                elif y_train[i] == "spam" and (word in emails_train[i]):
                    spam_counter_w += 1
            Pwcham.append((ham_counter_w + 1)/ (ham_counter + 2))
            Pwcspam.append((spam_counter_w + 1) / (spam_counter + 2))
        for x in Pwcham:
            Pdcham = Pdcham * x
        for x in Pwcspam:
            Pdcspam = Pdcspam * x

        if (Pdcham * Pcham) > (Pdcspam * Pcspam):
            y_pred.append("ham")
        else:
            y_pred.append("spam")

    correct = 0
    for i in range(0, len(y_test)):
        if y_test[i] == y_pred[i]:
            correct += 1

    print("Accuracy: ", correct / len(y_test))