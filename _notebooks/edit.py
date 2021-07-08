#!/usr/bin/python

import sys, re, datetime, os



def edit():
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    old_filename = str(sys.argv[1])
    new_filename = today_str+"-"+old_filename
    
    path = "/home/matt/Documents/github_page/mwcoleman.github.io/_posts/" 
    yaml_args = sys.argv
    yaml_args.extend(("true", "true", "true", "true"))  # If we don't pass any yaml in, do default
    yaml = f"---\ntitle: {yaml_args[1]}\nmathjax: {yaml_args[2]}\ncategories:\n  - {yaml_args[3]}\ntags:\n  - {yaml_args[4]}\n---\n\n"
    with open(path+old_filename, 'r') as file:
        filedata = file.read()
    filedata = re.sub(r"!\[png\]\(", "<img src=\"/mages/", filedata)
    filedata = re.sub(".png\)", ".png\">", filedata)
    filedata = yaml + filedata
    with open(path+old_filename, 'w') as file:
        file.write(filedata)
    # Name it correctly for jekyll
    os.rename(path+old_filename, path+new_filename) 

if __name__ == '__main__':
    edit()
