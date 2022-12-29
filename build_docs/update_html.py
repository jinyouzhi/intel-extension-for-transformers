import os, sys


def main(folder, version):
    index_file = "{}/index.html".format(folder)
    index_buf = ""
    folder_name=os.path.basename(folder)
    with open(index_file, "r") as f:
        index_buf = f.read()
        key_str='  <div class="version">\n                {}\n              </div>'.format(version)
        version_list = '''<div class="version">
              <a href="../versions.html">{}▼</a>
              <p>Click link above to switch version</p>
            </div>'''.format(folder_name)
        #print(index_buf.find(key_str))
        index_buf = index_buf.replace(key_str, version_list)
        #print(index_buf)

    with open(index_file, "w") as f:
        f.write(index_buf)

    version_file = "{}/versions.html".format(os.path.dirname(folder))
    #print(version_file)
    ver_buf = ""
    with open(version_file, "r") as f:
        ver_buf = f.read()
        if ver_buf.find(version)>=0:
            return
        key_str = '<li><a href="latest">latest</a></li>'
        new_ver = '''<li><a href="latest">latest</a></li>
        <li><a href="{}">{}</a></li>'''.format(version, version)
        ver_buf = ver_buf.replace(key_str, new_ver)

    with open(version_file, "w") as f:
        f.write(ver_buf)

def help(me):
    print("python {} html_folder version".format(me))

if __name__=="__main__":
    if len(sys.argv)<3:
        help(sys.argv[0])
        sys.exit(1)

    folder = sys.argv[1]
    version = sys.argv[2]
    main(folder, version)