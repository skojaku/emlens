# emlens
A lightweight toolbox for analyzing embedding space

### APIs
See [here](doc/apis.md).

### When adding your tools
- Write a docscring in a standard format (like numpydoc and others)
- Please update doc/apis.md (there is a script to automate this process)

### Update apis.md
Under the repository root directory, run 

```
bash scripts/generate-doc.sh
``` 
which will automatically generate doc/apis.md. `pdoc` is needed to run this script.
