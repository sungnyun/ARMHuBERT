OUTPUT_DIR="armhubert"  # change

cp -r ./modules ../s3prl/s3prl/upstream/$OUTPUT_DIR
cp ./$OUTPUT_DIR/expert.py ../s3prl/s3prl/upstream/$OUTPUT_DIR/
cp ./$OUTPUT_DIR/hubconf.py ../s3prl/s3prl/upstream/$OUTPUT_DIR/
