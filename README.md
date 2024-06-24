## X-Gear: Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction

Code for our ACL-2022 paper [Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction](https://arxiv.org/abs/2203.08308).


### Setup 

  - Python=3.7.10
  ```
  $ conda env create -f environment.yml
  ```

### Data and Preprocessing

- Go into the folder `./preprocessing/`
- If you follow the instruction in the README.md, then you can get your data in the folder `./processed_data/`

### Training

- Run `./scripts/generate_data_ace05.sh` and `./scripts/generate_data_ere.sh` to generate training examples of different languages for X-Gear. 
  The generated training data will be saved in `./finetuned_data/`.
- Run `./scripts/train_ace05.sh` or `./scripts/train_ere.sh` to train X-Gear. Alternatively, you can run the following command.

  ```
  python ./xgear/train.py -c ./config/config_ace05_mT5copy-base_en.json
  ```
  
  This trains X-Gear with mT5-base + copy mechanisim for ACE-05 English. The model will be saved in `./output/`.
  You can modify the arguments in the config file or replace the config file with other files in `./config/`.
  
### Evaluating

- Run the following script to evaluate the performance for ACE-05 English, Arabic, and Chinese.

  ```
  ./scripts/eval_ace05.sh [model_path] [prediction_dir]
  ```
  
  If you want to test X-Gear with mT5-large, remember to modify the config file in `./scripts/eval_ace05.sh`.
  
- Run the following script to evaluate the performance for ERE English and Spanish.

  ```
  ./scripts/eval_ere.sh [model_path] [prediction_dir]
  ```
  
  If you want to test X-Gear with mT5-large, remember to modify the config file in `./scripts/eval_ere.sh`.
  
We provide our pre-trained models and show their performances as follows.

**ACE-05**
|                          | en Arg-I | en Arg-C | ar Arg-I | ar Arg-C | zh Arg-I | zh Arg-C |
|--------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [X-Gear-ace05-mT5-base+copy-en](https://drive.google.com/file/d/1TyPlEc3nvbnU3Qj5IC_osyIXTGdK8Gkc/view?usp=sharing)  |   73.39  |   69.28  |   47.64  |   42.09  |   57.81  |   54.46  |
| [X-Gear-ace05-mT5-base+copy-ar](https://drive.google.com/file/d/15uDCNTdN_jXa51vxnJ6XGVpDLzvhBGmJ/view?usp=sharing)  |   33.87  |   27.17  |   72.97  |   66.92  |   31.14  |   28.84  |
| [X-Gear-ace05-mT5-base+copy-zh](https://drive.google.com/file/d/1o9OUKi6VWj1isxXHIQd6d9RGdhKLtklI/view?usp=sharing)  |   59.85  |   55.15  |   38.04  |   34.88  |   72.93  |   68.99  |
| [X-Gear-ace05-mT5-large+copy-en](https://drive.google.com/file/d/1690108Q-7P4F0Q_LUk3uH4nhwCoKtKlP/view?usp=sharing) |   75.16  |   71.85  |   54.18  |   50.00  |   63.14  |   58.40  |
| [X-Gear-ace05-mT5-large+copy-ar](https://drive.google.com/file/d/1U8b4Yhqbmw_T763q_09TEO5jRZftffnm/view?usp=sharing) |   38.81  |   34.57  |   73.49  |   67.75  |   39.26  |   36.13  |
| [X-Gear-ace05-mT5-large+copy-zh](https://drive.google.com/file/d/1NX1IIyWYqZNPRm1u2qum70MEpXhiqifZ/view?usp=sharing) |   61.44  |   55.40  |   38.71  |   36.14  |   70.45  |   66.99  |

**ERE**
|                              | en Arg-I | en Arg-C | es Arg-I | es Arg-C |
|------------------------------|:--------:|:--------:|:--------:|:--------:|
| [X-Gear-ere-mT5-base+copy-en](https://drive.google.com/file/d/1Wfl4AZHeI7YQaqZfg3PQcim17bpbU8l9/view?usp=sharing)  |   78.26  |   71.55  |   64.31  |   58.70  |
| [X-Gear-ere-mT5-base+copy-es](https://drive.google.com/file/d/1xKDMha62bdg5ZNol3UrTEEMqLXlG96CF/view?usp=sharing)  |   69.21  |   59.79  |   70.67  |   66.37  |
| [X-Gear-ere-mT5-large+copy-en](https://drive.google.com/file/d/1E2u5dHyF5s9tGF1vD47L3MJEkhnxkVUa/view?usp=sharing) |   78.10  |   73.04  |   64.82  |   60.35  |
| [X-Gear-ere-mT5-large+copy-es](https://drive.google.com/file/d/1X_LfrPsf6Ysy7T7WTcIbACWUTrxPuqHQ/view?usp=sharing) |   69.03  |   63.73  |   71.47  |   68.49  |

### Citation

If you find that the code is useful in your research, please consider citing our paper.

    @inproceedings{acl2022xgear,
        author    = {Kuan-Hao Huang and I-Hung Hsu and Premkumar Natarajan and Kai-Wei Chang and Nanyun Peng},
        title     = {Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction},
        booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
        year      = {2022},
    }
