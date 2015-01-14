import csv, os, re

# parse input data into different score files
train_input = os.path.join('data', 'raw', 'train.tsv')
train_output_score0 = os.path.join('data', 'processed_score', 'score0.txt')
train_output_score1 = os.path.join('data', 'processed_score', 'score1.txt')
train_output_score2 = os.path.join('data', 'processed_score', 'score2.txt')
train_output_score3 = os.path.join('data', 'processed_score', 'score3.txt')
train_output_score4 = os.path.join('data', 'processed_score', 'score4.txt')

#process tsv file
def convert_format(data_input):
  # print "\nReading:",data_input,"\nWriting:",data_output
  with open(data_input, 'rb') as infile, open(train_output_score0, "wb") as outfile0, open(train_output_score1, "wb") as outfile1, open(train_output_score2, "wb") as outfile2, open(train_output_score3, "wb") as outfile3, open(train_output_score4, "wb") as outfile4:
    reader = csv.DictReader(infile, delimiter="\t", quotechar='"')
    for row in reader:
      label = int(row['Sentiment'])
      if label == 0:
        outfile0.write(row['Phrase'] + "\n")
      elif label == 1:
        outfile1.write(row['Phrase'] + "\n")
      elif label == 2:
        outfile2.write(row['Phrase'] + "\n")
      elif label == 3:
        outfile3.write(row['Phrase'] + "\n")
      elif label == 4:
        outfile4.write(row['Phrase'] + "\n")
          
convert_format(train_input)