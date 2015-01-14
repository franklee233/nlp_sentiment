import csv, os, re

# parse input data into different score files
test_input = os.path.join('data', 'raw', 'train.tsv')
test_output = os.path.join('data', 'processed_score', 'test.txt')

#process tsv file
def convert_format(data_input):
  # print "\nReading:",data_input,"\nWriting:",data_output
  with open(test_input, 'rb') as infile, open(test_output, "wb") as outfile:
    reader = csv.DictReader(infile, delimiter="\t", quotechar='"')
    for row in reader:
      label = int(row['Sentiment'])
      outfile.write(row['Phrase'] + "\n")
          
convert_format(test_input)