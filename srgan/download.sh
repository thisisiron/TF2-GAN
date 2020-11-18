# ref: https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download

mkdir data

# 512x512
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1E23HCNL-v9c54Wnzkm9yippBW8IaLUXp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E23HCNL-v9c54Wnzkm9yippBW8IaLUXp" -O data512x512.zip && rm -rf /tmp/cookies.txt

mv data512x512.zip ./data
unzip ./data/data512x512.zip
