# добавляем vscode в PATH
PATH=“$PATH:$HOME/.vscode/bin”
 
# цветовая настройка
export CLICOLOR=1
export LSCOLORS=GxFxCxDBxegedabagaced
export PS1=“\[\e[32m\]\u@h \[\e[34m\]\w\[\e[0m\] $ “
 
# alias-ы
alias ls=‘ls —color=auto’
alias grep=‘grep —color=auto’
alias fgrep=‘fgrep —color=auto’
alias egrep=‘egrep —color=auto’
alias cp=‘cp -i’
alias df=‘df -h’
alias free=‘free -m’
 
# alias-ы для du
alias du-mb=‘du -ah —block-size=M | sort -hr’
alias du-gb=‘du -ah —block-size=G | sort -hr’
alias du-dir-mb=‘du -sh —block-size=M */ | sort -hr’
alias du-dir-gb=‘du -sh —block-size=G */ | sort -hr’
 
# команда для запуска
alias loadrc=‘source /home/datalab/nfs/.bashrc’
 
# кастомные настройки vscode
# alias code=“code —user-data-dir=/home/datalab/nfs/.vscode”
 
export PIP_CONFIG_FILE=“/home/datalab/nfs/.config/pip/pip.conf”
 
# install faiss and annoy every time
pip install —no-cache-dir annoy faiss-cpu