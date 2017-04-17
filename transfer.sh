mkdir -p to_aws
rm -r to_aws/*
cp  data.tar.gz to_aws/.
cp -r *.py to_aws/.
scp -r to_aws carnd@$1:.
