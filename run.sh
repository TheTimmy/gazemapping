#!/bin/sh
for cubemap in 0 1 2 3 4
do
    clear && ./VocabTree --command desc --config "../Cubemap-000$i.json"
    clear && ./VocabTree --command tree --config "../Cubemap-000$i.json"
    clear && ./VocabTree --command fixation --config "../Cubemap-000$i.json"
done
