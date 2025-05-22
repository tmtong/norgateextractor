serve:
	uvicorn norgateextractor.server:app --host 0.0.0.0 --port 8000 --loop asyncio --http httptools --reload

requirements:
	chmod 700 sec
	pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p norgatedata/index norgatedata/stock
	

gitrcommit:
	git config http.postBuffer 524288000
	git config ssh.postBuffer 524288000
	rm -rf .git/hooks/pre-push .git/hooks/post-push
	rm -rf .git/hooks/post-commit
	rm -rf .git/hooks/post-merge .git/hooks/post-checkout
	git config --global http.sslVerify false
	git config --global credential.helper store
	# git add -u
	git add Makefile requirements.txt
	git add norgateextractor
	-git commit -a -m "`date`"
	git pull --no-rebase
	git push origin HEAD

gitrupdate:
	rm -rf .git/hooks/pre-push .git/hooks/post-push
	rm -rf .git/hooks/post-commit
	rm -rf .git/hooks/post-merge .git/hooks/post-checkout
	git config --global http.sslVerify false
	git config --global credential.helper store
	git pull --no-rebase

copyz:
	cp -vr norgatedata/* /cygdrive/z/
