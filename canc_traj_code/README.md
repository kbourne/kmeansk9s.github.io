# CANCER TRAJECTORY ANALYSIS FOR CANCER IN DOGS - CODE

Masters of Applied Data Science Capstone Project

`CANCER TRAJECTORY ANALYSIS FOR CANCER IN DOGS` focused on exploring the use of machine learning in predicting cancer and its trajectory among companion dogs.

<img src="../images/victor-grabarczyk-x5oPmHmY3kQ-unsplash.jpg" width="340" align="center"> <img src="../images/taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg" width="286" align="center"> <img src="../images/t-r-photography-TzjMd7i5WQI-unsplash.jpg" width="286" align="center"> 

Contents
========

 * [Why?](#why)
 * [What is in this repo?](#what-is-in-this-repo)
 * [What do I need to run this code?](#what-do-I-need-to-run-this-code)

### Why?

It is widely recognized in the veterinary world that dogs provide a unique model for health research that parallels the human environment. Dogs are exposed to similar social and environmental elements as humans, exhibiting increases in many chronic conditions with dynamics similar to human patterns¹. Dogs also have shorter life spans, which allows researchers to observe their entire life course in a much more condensed time frame². Use of machine learning in human healthcare has advanced rapidly in recent years, paving the way for new and deeper insights into how data can be used to improve human healthcare. Due to the similarities between human and dog healthcare, we seek to bring these analytical innovations to dog healthcare, with the hopes of finding deeper insights that can help both canine and human care. This analysis begins a number of traditional machine learning analysis techniques applied to the dataset. It will conclude with the application of two cutting edge techniques that have emerged in human healthcare, but applied to this dog healthcare set as a way to determine how new techniques in a similar field can help this field advance. And last, we should consider the ethical implications of the data obtained from all of these owners and dogs, particularly when it comes to privacy.

### What is in this repo?
---

In this repo are Jupyter Notebooks that run:

+ Back up dotfiles _from where they live on the system_.
+ Back up files from _any_ path on the system, not just `$HOME`.
+ Reinstall them from the backup directory idempotently.
+ Backup and reinstall files conditionally, so you can easily manage dotfiles across multiple systems.
+ Copy files on installation and backup, as opposed to symlinking them.
+ Backup package installations in a highly compressed manner

There are also python files that are used by the Notebooks that contain most of the model-related 

And is incredibly fault tolerant and user-protective.

### What do I need to run this code?
---

To start the interactive program, simply run `$ shallow-backup`.

`shallow-backup` was built with scripting in mind. Every feature that's supported in the interactive program is supported with command line arguments.

```shell
Usage: shallow-backup [OPTIONS]

  Easily back up installed packages, dotfiles, and more.
  You can edit which files are backed up in ~/.shallow-backup.

  Written by Aaron Lichtman (@alichtman).

Options:
  --add-dot TEXT              Add a dotfile or dotfolder to config by path.
  -backup-all                 Full back up.
  -backup-configs             Back up app config files.
  -backup-dots                Back up dotfiles.
  -backup-packages            Back up package libraries.
  -delete-config              Delete config file.
  -destroy-backup             Delete backup directory.
  -dry-run                    Don't backup or reinstall any files, just give
                              verbose output.

  -backup-fonts               Back up installed fonts.
  --new-path TEXT             Input a new back up directory path.
  -no-new-backup-path-prompt  Skip setting new back up directory path prompt.
  -no-splash                  Don't display splash screen.
  -reinstall-all              Full reinstallation.
  -reinstall-configs          Reinstall configs.
  -reinstall-dots             Reinstall dotfiles and dotfolders.
  -reinstall-fonts            Reinstall fonts.
  -reinstall-packages         Reinstall packages.
  --remote TEXT               Set remote URL for the git repo.
  -separate-dotfiles-repo     Use if you are trying to maintain a separate
                              dotfiles repo and running into issue #229.

  -show                       Display config file.
  -v, --version               Display version and author info.
  -h, -help, --help           Show this message and exit.
```


1. Select the appropriate option in the CLI and follow the prompts.
2. Open the file in a text editor and make your changes.


Here's an example config based on my [dotfiles](https://www.github.com/alichtman/dotfiles):

```json
{
	"backup_path": "~/shallow-backup",
	"lowest_supported_version": "5.0.0a",
	"dotfiles": {
		".config/agignore": {
			"backup_condition": "uname -a | grep Darwin",
			"reinstall_conditon": "uname -a | grep Darwin"
		},
		".config/git/gitignore_global": { },
		".config/jrnl/jrnl.yaml": { },
		".config/kitty": { },
		".config/nvim": { },
		".config/pycodestyle": { },
		...
		".zshenv": { }
	},
	"root-gitignore": [
		".DS_Store",
		"dotfiles/.config/nvim/.netrwhist",
		"dotfiles/.config/nvim/spell/en.utf-8.add",
		"dotfiles/.config/ranger/plugins/ranger_devicons",
		"dotfiles/.config/zsh/.zcompdump*",
		"dotfiles/.pypirc",
		"dotfiles/.ssh"
	],
	"dotfiles-gitignore": [
		".DS_Store",
		".config/nvim/.netrwhist",
		".config/nvim/spell/en.utf-8.add*",
		".config/ranger/plugins/*",
		".config/zsh/.zcompdump*",
		".config/zsh/.zinit",
		".config/tmux/plugins",
		".config/tmux/resurrect",
		".pypirc",
		".ssh/*"
	],
	"config_mapping": {
		"/Users/alichtman/Library/Application Support/Sublime Text 2": "sublime2",
		"/Users/alichtman/Library/Application Support/Sublime Text 3": "sublime3",
		"/Users/alichtman/Library/Application Support/Code/User/settings.json": "vscode/settings",
		"/Users/alichtman/Library/Application Support/Code/User/Snippets": "vscode/Snippets",
		"/Users/alichtman/Library/Application Support/Code/User/keybindings.json": "vscode/keybindings",
		"/Users/alichtman/.atom": "atom",
		"/Users/alichtman/Library/Preferences/com.apple.Terminal.plist": "terminal_plist"
	}
}
```
 The original `default-gitignore` key in the config is still supported for backwards compatibility, however, converting to the new config format is strongly encouraged.

#### Output Structure
---

```shell
backup_dir/
├── configs
│   ├── plist
│   │   └── com.apple.Terminal.plist
│   ├── sublime_2
│   │   └── ...
│   └── sublime_3
│       └── ...
├── dotfiles
│   ├── .bash_profile
│   ├── .bashrc
│   ├── .gitconfig
│   ├── .pypirc
│   ├── ...
│   ├── .shallow-backup
│   ├── .ssh/
│   │   └── known_hosts
│   ├── .vim/
│   └── .zshrc
├── fonts
│   ├── AllerDisplay.ttf
│   ├── Aller_Bd.ttf
│   ├── ...
│   ├── Ubuntu Mono derivative Powerline Italic.ttf
│   └── Ubuntu Mono derivative Powerline.ttf
└── packages
    ├── apm_list.txt
    ├── brew-cask_list.txt
    ├── brew_list.txt
    ├── cargo_list.txt
    ├── gem_list.txt
    ├── installed_apps_list.txt
    ├── npm_list.txt
    ├── macports_list.txt
    ├── pip_list.txt
    └── sublime3_list.txt
```



> **Warning**
> Be careful running this with elevated privileges. Code execution can be achieved with write permissions on the config file.


#### Method 1: [`pip3`](https://pypi.org/project/shallow-backup/)

```bash
$ pip3 install shallow-backup
```

#### Method 2: Install From Source

```bash
$ git clone https://www.github.com/alichtman/shallow-backup.git
$ cd shallow-backup
$ pip3 install .
```