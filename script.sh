#!/bin/bash

set -x
rm -r in
rm -r out
mkdir -p in
mkdir -p out
ls

cd sssp
make
K=( 3 4 6 8)
S=( 4 5 6 7)
for i in "${!S[@]}"
  do
  pwd
  ./gen_RMAT -s "${S[i]}" -k "${K[i]}" -nRoots 1 -out "../in/rmat-"${S[i]}".txt"
  ./graphs_reference -in "../in/rmat-"${S[i]}".txt" -root 0 -out "../out/rmat-"${S[i]}"-serial.txt"
done
cd ..
for i in "${S[@]}"
do
  if [ ${i} = "4" ]; then
    python3 -m charmrun.start +p2 main.py -i "./in/rmat-${i}.txt" -o "./out/rmat-"${i}"-parallel.txt" -r 0 -d
  else # no debug output
    python3 -m charmrun.start +p2 main.py -i "./in/rmat-${i}.txt" -o "./out/rmat-"${i}"-parallel.txt" -r 0
  fi
done

cd ./sssp
for i in "${S[@]}"
do
  ./compare "../out/rmat-${i}-serial.txt" "../out/rmat-"${i}"-parallel.txt"
done

cd ./sssp
make clean
cd ../
