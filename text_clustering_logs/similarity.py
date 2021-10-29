# from https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
# paper: https://arxiv.org/pdf/1908.10084.pdf

import requests
import os

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-xlm-r-multilingual-v1"
headers = {"Authorization": "Bearer %s" % os.environ['HUGGINGFACE_API_KEY']}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()




output = query({
    "inputs": {
		"source_sentence": "That is a happy person",
		"sentences": [
			"That is a happy dog",
			"That is a very happy person",
			"Today is a sunny day"
		]
	},
})



print(output)



log = """
localhost - - [09/Mar/2021:12:56:51 +1100] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [21/Mar/2021:15:11:06 +1100] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1311 Create-Job successful-ok
localhost - - [21/Mar/2021:15:11:06 +1100] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1394704 Send-Document successful-ok
localhost - - [21/Mar/2021:15:11:06 +1100] "POST / HTTP/1.1" 200 353 Set-Job-Attributes successful-ok
localhost - - [21/Mar/2021:15:11:34 +1100] "POST / HTTP/1.1" 200 1394551 CUPS-Get-Document successful-ok
localhost - - [21/Mar/2021:16:54:47 +1100] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:16:54:48 +1100] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [21/Mar/2021:21:07:01 +1100] "POST /admin HTTP/1.1" 401 234 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:21:07:01 +1100] "POST /admin HTTP/1.1" 200 234 CUPS-Add-Modify-Printer successful-ok
localhost - - [21/Mar/2021:21:10:28 +1100] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:21:10:28 +1100] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [21/Mar/2021:21:33:56 +1100] "POST /admin HTTP/1.1" 401 234 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:21:33:56 +1100] "POST /admin HTTP/1.1" 200 234 CUPS-Add-Modify-Printer successful-ok
localhost - - [21/Mar/2021:21:37:36 +1100] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:21:37:36 +1100] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [21/Mar/2021:21:50:29 +1100] "POST /admin HTTP/1.1" 401 234 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [21/Mar/2021:21:50:29 +1100] "POST /admin HTTP/1.1" 200 234 CUPS-Add-Modify-Printer successful-ok
localhost - - [22/Mar/2021:10:34:13 +1100] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [22/Mar/2021:10:34:14 +1100] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [03/Apr/2021:12:53:18 +1100] "POST /admin HTTP/1.1" 401 238 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [03/Apr/2021:12:53:18 +1100] "POST /admin HTTP/1.1" 200 238 CUPS-Add-Modify-Printer successful-ok
localhost - - [03/Apr/2021:12:53:22 +1100] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 4436 Create-Job successful-ok
localhost - - [03/Apr/2021:12:53:22 +1100] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 105710 Send-Document successful-ok
localhost - - [03/Apr/2021:12:53:22 +1100] "POST / HTTP/1.1" 200 353 Set-Job-Attributes successful-ok
localhost - - [03/Apr/2021:12:53:31 +1100] "POST / HTTP/1.1" 200 105560 CUPS-Get-Document successful-ok
localhost - - [03/Apr/2021:15:58:45 +1100] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [03/Apr/2021:15:58:45 +1100] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [06/Apr/2021:08:27:48 +1000] "POST /admin HTTP/1.1" 401 234 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [06/Apr/2021:08:27:48 +1000] "POST /admin HTTP/1.1" 200 234 CUPS-Add-Modify-Printer successful-ok
localhost - - [03/May/2021:14:29:35 +1000] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [07/May/2021:15:31:08 +1000] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [08/Jun/2021:22:28:02 +1000] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [28/Jun/2021:09:27:41 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1225 Create-Job successful-ok
localhost - - [28/Jun/2021:09:27:41 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 67924 Send-Document successful-ok
localhost - - [28/Jun/2021:09:27:41 +1000] "POST / HTTP/1.1" 200 353 Set-Job-Attributes successful-ok
localhost - - [28/Jun/2021:09:28:38 +1000] "POST / HTTP/1.1" 200 67785 CUPS-Get-Document successful-ok
localhost - - [28/Jun/2021:09:30:45 +1000] "POST /jobs HTTP/1.1" 200 147 Cancel-Job successful-ok
localhost - - [28/Jun/2021:09:30:55 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1225 Create-Job successful-ok
localhost - - [28/Jun/2021:09:30:55 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 67912 Send-Document successful-ok
localhost - - [28/Jun/2021:09:30:55 +1000] "POST / HTTP/1.1" 200 353 Set-Job-Attributes successful-ok
localhost - - [28/Jun/2021:09:30:59 +1000] "POST / HTTP/1.1" 200 67773 CUPS-Get-Document successful-ok
localhost - - [28/Jun/2021:09:32:16 +1000] "POST /admin HTTP/1.1" 401 253 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [28/Jun/2021:09:32:16 +1000] "POST /admin HTTP/1.1" 200 253 CUPS-Add-Modify-Printer successful-ok
localhost - - [02/Jul/2021:10:33:05 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1213 Create-Job successful-ok
localhost - - [02/Jul/2021:10:33:05 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 87018 Send-Document successful-ok
localhost - - [02/Jul/2021:10:33:05 +1000] "POST / HTTP/1.1" 200 353 Set-Job-Attributes successful-ok
localhost - - [02/Jul/2021:14:41:26 +1000] "POST / HTTP/1.1" 200 86900 CUPS-Get-Document successful-ok
localhost - - [23/Aug/2021:17:37:19 +1000] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [16/Sep/2021:11:11:08 +1000] "POST / HTTP/1.1" 200 69508 CUPS-Get-PPDs -
localhost - - [17/Sep/2021:13:11:30 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 1564 Create-Job successful-ok
localhost - - [17/Sep/2021:13:11:30 +1000] "POST /printers/Samsung_M2070_Series__SEC30CDA7AFE250_ HTTP/1.1" 200 199635 Send-Document successful-ok
localhost - - [17/Sep/2021:13:11:30 +1000] "POST / HTTP/1.1" 200 319 Set-Job-Attributes successful-ok
localhost - - [17/Sep/2021:13:11:31 +1000] "POST / HTTP/1.1" 200 199504 CUPS-Get-Document successful-ok
localhost - - [17/Sep/2021:17:56:12 +1000] "POST /admin HTTP/1.1" 401 232 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:17:56:13 +1000] "POST /admin HTTP/1.1" 200 232 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:17:57:14 +1000] "POST /admin HTTP/1.1" 200 222 CUPS-Add-Modify-Printer successful-ok
localhost - - [17/Sep/2021:17:59:42 +1000] "POST /admin HTTP/1.1" 401 232 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:17:59:42 +1000] "POST /admin HTTP/1.1" 200 232 CUPS-Add-Modify-Printer successful-ok
localhost - - [17/Sep/2021:18:00:57 +1000] "POST /admin HTTP/1.1" 401 222 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:18:00:57 +1000] "POST /admin HTTP/1.1" 200 222 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:18:03:46 +1000] "POST /admin HTTP/1.1" 200 232 CUPS-Add-Modify-Printer successful-ok
localhost - - [17/Sep/2021:18:39:13 +1000] "POST /admin HTTP/1.1" 401 222 CUPS-Add-Modify-Printer successful-ok
localhost - michaelneale [17/Sep/2021:18:39:13 +1000] "POST /admin HTTP/1.1" 200 222 CUPS-Add-Modify-Printer successful-ok
localhost - - [18/Sep/2021:05:52:36 +1000] "POST /admin HTTP/1.1" 401 232 CUPS-Add-Modify-Printer successful-ok
"""

logs = log.split("\n")
output = query({
    "inputs": {
		"source_sentence": '[G2MUpload] <main> Nothing to send',
		"sentences": logs
	},
})

print(output)

highest_match = max(output)
print(highest_match)

if highest_match < 0.4:
	print("WE HAVE AN ANOMALY")


def semantic_similarity():
	# from https://www.sbert.net/docs/usage/semantic_textual_similarity.html

	from sentence_transformers import SentenceTransformer, util
	sentences = ["This is an example sentence", "Each sentence is converted"]

	model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')


	# Two lists of sentences

	sentences1 = ['The dog plays in the garden',]

	sentences2 = ['The cat sits outside',
				'A man is playing guitar',
				'The new movie is awesome']


	#Compute embedding for both lists
	embeddings1 = model.encode(sentences1, convert_to_tensor=True)
	embeddings2 = model.encode(sentences2, convert_to_tensor=True)

	#Compute cosine-similarits
	cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

	print(max(cosine_scores[0]))
	#Output the pairs with their score
	#for i in range(len(sentences1)):
	#   print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))