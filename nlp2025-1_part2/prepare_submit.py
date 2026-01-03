# 과제 평가를 위해 zip 파일을 만들어 제출합니다.

import os
import zipfile

required_files = [p for p in os.listdir('.') if p.endswith('.py')] + \
                 [f'predictions/{p}' for p in os.listdir('predictions')] + \
                     [f'models/{p}' for p in os.listdir('models')] + \
                        [f'modules/{p}' for p in os.listdir('modules')]

def main():
    aid = 'nlp2025-1_project_outputs'

    with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()
