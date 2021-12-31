## Welcome to Auto Coder

This is a suite of APIs to help speed up developers day to day lives with an AI assistant, powered by GPT-3 and google cloud functions.

I aim to integrate much of this with IDEs like VS code (some is already) and github pull request workflows and more. 
Here the apis are documented that you can try out.

### Explain code you don't understand

With this API you provide a chunk of code, and auto coder will explain what it does in natural language. For example:

> ```curl --request POST --header 'Content-Type: application/json' --url https://us-east1-micprojects.cloudfunctions.net/ai-code-explainer --data '{"code":"for i in range(10):\n    print(i)\n"}'```

This will explain the code as best it can with the context given and its knowledge. You can include any existing comments and context you like. 

> The code prints out the numbers from 0 to 9.

### Look at code for security issues 

The AI can look for security problems you can accidentally slip in. For example: 

> ```curl --request POST --header 'Content-Type: application/json' --url https://us-east1-micprojects.cloudfunctions.net/ai-security-scan  -d '{"code": "class Log:\ndef __init__(self, path):\n        dirname = os.path.dirname(path)\n       os.makedirs(dirname, exist_ok=True)\n "}'```

Will return: 

>  The code is vulnerable to directory traversal.

Yeah probably should fix that!

### Generate unit tests. 

Ever said "I'll write some unit tests when I get time" - well the auto coder has your back: 

For a trivial example: 

> ```curl --request POST --header 'Content-Type: application/json' --url https://us-east1-micprojects.cloudfunctions.net/ai-auto-test --data '{"code":"def calc(a,b):\n    return a+b\n"}'```

```
def test_calc():
    assert calc(1,2) == 3   
```

It works for multiple languages and more complex, less contrived snippets of code too (give it a try). 


### Remove old feature flags from your code

Feature flags are if statements that litter your code over time. Eventually they end up all turned on, so you need to go back and remove them. 
Try using auto coder to help you out: 

> ```curl --data-binary @src/services/Rox.ts "https://us-east1-micprojects.cloudfunctions.net/auto-flag-remover?flag_name=showDeleteExperimentButtonInNewFlagsView" -o src/services/Rox.ts```

This will update one file at a time, but if you are using github, you can use a Personal Access Token and have it update your org all in one hit, opening pull requests for each repository that needs a change: 

> ```https://us-east1-micprojects.cloudfunctions.net/auto-flag-remover?flag_name=showDeleteExperimentButtonInNewFlagsView&org=michaelneale&token=YOUR_GITHUB_PERSONAL_ACCESS_TOKEN```

### More to come!







### more complex examples: 

> ```curl --request POST --header 'Content-Type: application/json' --url https://us-east1-micprojects.cloudfunctions.net/ai-code-explainer --data '{"code":"    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,\ncallbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)\nprint(model.evaluate(X_test, y_test))"}'```
