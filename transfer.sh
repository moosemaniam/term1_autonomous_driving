mkdir to_aws
cp -r data to_aws/.
cp -r *.py to_aws/.
tar -zcvf to_aws.tar.gz to_aws
scp to_aws.tar.gz carnd@$1:.
