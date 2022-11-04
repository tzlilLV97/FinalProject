
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from SysEvalOffTarget_src import general_utilities

from Bio import SeqIO
input_file = general_utilities.HOME_DIR + "GRCh38_latest_genomic.fna"
output_file = "GRCH38222222.txt"
genom = ""
fasta_sequences = SeqIO.parse(open(input_file),'fasta')
i = 0
genom_array = [0 for i in range(25)]
print("kaki")
dataset_df = pd.read_excel(general_utilities.CHANGE_SEQ_PATH)
labels = []
print(dataset_df.columns)
#moshiko = SeqIO.read(input_file,"fasta")
if 0:
    with open(output_file,'w') as out_file:
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            genom += sequence
            if name[0:7]=="NC_0000":
                temp = int(name[7:9])
                genom_array[temp] = sequence
            #    genom += sequence
            #    print(len(genom))
          #  print(name)
            if i==26:
                break
            if name[0:2]=="NC":
                i+=1
            if name[0:2] not in labels:
                labels.append(name[0:2])
             #   print(genom)
               #break
          #  print(name)

#print(len(genom))
#print(genom)
print(labels)
print("What got parsed : ")
print("Off target : CACACTAATCCTGTCCCCAGAGG")
print("Target : GTCACCAATCCTGTCCCTAGNGG")
print(genom[12523844:12523867])
print("What got parsed : ")
print("Off target : GGCACCAATGCTGTCCTACCAGG")
print("Target : GTCACCAATCCTGTCCCTAGNGG")
print(genom[5332528:5332551])

