import csv, os, re

# parse input data into different 2 files
train_input = os.path.join('data', 'raw', 'train.tsv')
test_input = os.path.join('data', 'raw', 'test.tsv')
train_output = "train_processed"
test_output = "test_processed"

#parse the sentence into words
def parse(s):
  return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()

#process tsv file
def convert_format(data_input, data_output, testFlag = False):
  print "\nReading:",data_input,"\nWriting:",data_output
  with open(data_input, 'rb') as infile, open(data_output, "wb") as outfile:
    reader = csv.DictReader(infile, delimiter="\t", quotechar='"')
    for row in reader:
      if testFlag:
        label = "1"
      else:
        label = str(int(row['Sentiment'])+1)
      phrase = parse(row['Phrase'])
      outfile.write(   label + " '"+row['PhraseId'] + " |f " + phrase + " |a " + "word_count:"+str(phrase.count(" ")+1) + "\n" )
          
convert_format(train_input, train_output)
convert_format(test_input, test_output, testFlag=True)