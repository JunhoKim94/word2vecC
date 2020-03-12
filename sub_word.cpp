#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <io.h>
#include <math.h>
#include <time.h>

#include <process.h>
#include <windows.h>

#include <iostream>

using namespace std;

const int vocab_hash_size = 3000000;
const int ch_hash_size = 1000000;
#define MAX_SEN 1000
#define MAX_WORD_LEN 100
#define MAX_CODE_LEN 40
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define _CRT_SECURE_NO_WARNINGS
#define n_gram 3

struct vocab_word {
    long long freq;
    int* sub_word;
    char* word;
    unsigned long len;
};

struct char_word {
    char* word;
};


int min_freq = 5, win_var = 0, lr_decay = 0, verbose = 1, grad_clip = 1;
int skip = 5, negative_sampling = 5, embed_size = 300;
long long vocab_size = 0, vocab_max_size = 1000, train_words = 0, ch_size = 0, ch_max_size = 1000;
struct vocab_word* vocab;
struct char_word* sub;
int* vocab_hash, * ch_hash;
clock_t start;
int num_thread = 4;
char file_path[100][100];
int num = 100;
int epoch = 1;
float lr = 0.025;
float sub_sampling = 0.01;
float *Weight_emb, *NS_Weight, *expTable, *ch_Weight;
short s = 2;
const int table_size = 1e7;
int* uni_table;

void unigram_table()
{
    int a = 0, num;
    double total_p = 0, power = 0.75;
    double p;
    uni_table = (int*)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) total_p += pow(vocab[a].freq, power);

    num = 0;
    p = pow(vocab[num].freq, power) / total_p;
    for (a = 0; a < table_size; a++)
    {
        uni_table[a] = num;
        if (a / double(table_size) > p)
        {
            num++;
            p += pow(vocab[num].freq, power) / total_p;
        }
        if (num >= vocab_size) num = vocab_size - 1;
    }

}

void ReadWord(char* word, FILE* fp)
{
    //open 된 fp 와 buffer 수 만큼의 word --> 하나의 word 추출
    //" ", \n, \t 제외
    // NULL 값도 포함되어 있음 --> Word
    int ch;
    int i = 0;
    while (!feof(fp))
    {
        ch = fgetc(fp);
        //뺄거 더 없나 봐야댐
        if ((ch == ' ') || (ch == '\t') || (ch == '\n') || (ch == '-'))
        {
            if (ch == 13) continue;
            //문장 마지막 단어에 \n 붙어서 나오는것을 방지하기 위함
            if (i > 0)
            {
                if (ch == '\n') ungetc(ch, fp);
                break;
            }

            if (ch == '\n')
            {
                //문장 구분을 위한 추가
                strcpy(word, (char*)"</s>");
                return;
            }
            else continue;
        }
        word[i] = ch;
        i++;
        if (i >= MAX_WORD_LEN - 1) i--;

    }
    //종료문
    word[i] = '\0';
}

int GetHash(char* word)
{
    //word char 개수만큼 *257 + char 아스키
    unsigned long long i, hash = 0;
    for (i = 0; i < strlen(word); i++) hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;

    return hash;
}

int word2idx(char* word)
{
    //word가 vocab 과 vocab_hash에 존재하는지 return 하는 함수
    unsigned int hash = GetHash(word);
    while (1)
    {
        if (vocab_hash[hash] == -1) break;
        if (!strcmp(vocab[vocab_hash[hash]].word, word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int ReadWordIndex(FILE* fp)
{
    char word[MAX_WORD_LEN];
    ReadWord(word, fp);
    if (feof(fp)) return -1;
    return word2idx(word);
}

int Addword2vocab(char* word)
{
    //printf("%s  ", word);
    //word가 처음 들어왔을 때 vocab과 vocab_hash에 저장하는 함수
    unsigned int hash, length = strlen(word) + 1;

    if (length > MAX_WORD_LEN) length = MAX_WORD_LEN;
    //메모리가 해제되었는데 접근한 경우 segmentation fault
    vocab[vocab_size].word = (char*)calloc(length, sizeof(char));

    //해당 메모리에 복사
    strcpy(vocab[vocab_size].word, word);

    vocab[vocab_size].freq = 1;
    vocab_size++;
    if (vocab_size + 2 > vocab_max_size)
    {
        vocab_max_size += 1000;
        vocab = (struct vocab_word*)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }

    hash = GetHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;

    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

int VocabCompare(const void* a, const void* b)
{
    return ((struct vocab_word*)b)->freq - ((struct vocab_word*)a)->freq;
}

//to save memory Reduce vocab which have low freq
void Reducevocab()
{
    int a, b = 0;
    unsigned int hash;
    for (int a = 0; a < vocab_size; a++) if (min_freq > vocab[a].freq)
    {
        vocab[b].freq = vocab[a].freq;
        vocab[b].word = vocab[a].word;
        b++;
    }
    else free(vocab[a].word);

    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++)
    {
        hash = GetHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }

    printf("Reduce Word is Complete  |  Current Vocab_size = %ld \n", vocab_size);
}

void SortVocab()
{
    int i, size;
    unsigned int hash;
    //Huffman 사용하기 전 Vocab Sort 함수 & freq <min 이하는 제거
    //"UNK" 빼고 나머지 내림차순
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

    size = vocab_size;
    for (i = 0; i < size; i++)
    {
        //printf("%d   ", freq);
        if ((vocab[i].freq < min_freq) && (i != 0))
        {
            vocab_size--;
            free(vocab[i].word);
            free(vocab[i].sub_word);
        }

        else
        {
            hash = GetHash(vocab[i].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            // i 값이 연속적이게 들어가지 않음 : 문제 생김?
            vocab_hash[hash] = i;
        }
    }
    // 앞에서부터 짤림 --> 내림차순이라 줄어든 vocab_size 개수만큼 짜르면 알아서 cutting
    vocab = (struct vocab_word*)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));


    printf("final vocab size :: %lld \n", vocab_size);

}

int sub_word_Hash(char* ch)
{
    unsigned long long i, hash = 0;
    for (i = 0; i < strlen(ch); i++) hash = hash * 257 + ch[i];
    hash = hash % ch_hash_size;

    return hash;
}

int Add_sub(char* ch)
{
    //printf("%s  ", word);
    //word가 처음 들어왔을 때 vocab과 vocab_hash에 저장하는 함수
    unsigned int hash;

    //메모리가 해제되었는데 접근한 경우 segmentation fault
    sub[ch_size].word = (char*)calloc(strlen(ch) + 1, sizeof(char));

    //해당 메모리에 복사
    strcpy(sub[ch_size].word, ch);

    ch_size++;

    if (ch_size + 2 > ch_max_size)
    {
        ch_max_size += 1000;
        sub = (struct char_word*)realloc(sub, ch_max_size * sizeof(struct char_word));
    }
    hash = sub_word_Hash(ch);
    while (ch_hash[hash] != -1) hash = (hash + 1) % ch_hash_size;
    ch_hash[hash] = ch_size - 1;
    return ch_size - 1;
}

int ch2idx(char* ch)
{
    //word가 vocab 과 vocab_hash에 존재하는지 return 하는 함수
    unsigned int hash = sub_word_Hash(ch);
    while (1)
    {
        if (ch_hash[hash] == -1) break;
        if (!strcmp(sub[ch_hash[hash]].word, ch)) return ch_hash[hash];
        hash = (hash + 1) % ch_hash_size;
    }
    return -1;
}

void subword_corpus(char* word, int new_idx)
{
    char temp[MAX_WORD_LEN + 2];
    long long idx;
    char a, sum;
    char sub_word[n_gram + 1];
    temp[0] = '<';
    for (int i = 0; i < strlen(word); i++) temp[i + 1] = word[i];
    temp[strlen(word) + 1] = '>';
    temp[strlen(word) + 2] = '\0';

    sum = 0;
    //sum = (n_gram - s + 1) * (strlen(temp) + 1 - (s + n_gram) / 2);
    for (a = s; a < n_gram + 1; a++) if (strlen(temp) >= a) for (int i = 0; i < strlen(temp) - a + 1; i++) sum++;
    //structure 초기화는 항상 밖에서
    vocab[new_idx].sub_word = (int*)calloc(sum + 1, sizeof(int));
    vocab[new_idx].len = 0;
    for (int i = 0; i < sum + 1; i++) vocab[new_idx].sub_word[i] = -1;
    for (a = s ; a < n_gram + 1 ; a++) if (strlen(temp) >= a) for (int i = 0; i < strlen(temp) - a + 1; i++)
    {
        for (int j = 0; j < a; j++) sub_word[j] = temp[i + j];
        sub_word[a] = '\0';
        idx = ch2idx(sub_word);
        if (idx == -1) idx = Add_sub(sub_word);
            vocab[new_idx].sub_word[vocab[new_idx].len] = idx;
            vocab[new_idx].len++;

    }
    //단어가 n_gram 보다 더 길면 전체 단어 <> 추가
    if (strlen(temp) > n_gram)
    {
        idx = ch2idx(temp);
        if (idx == -1) idx = Add_sub(temp);
        vocab[new_idx].sub_word[vocab[new_idx].len] = idx;
        vocab[new_idx].len++;
    }

    //cout << strlen(temp) << "   " << temp << "   " << vocab[new_idx].word << "    " << vocab[new_idx].len << "    " << (int)sum << endl;
}

//vocab에 freq, path, word, id들 계산 --> path = Huffman // word == ReadWordID, id // freq
void Make_corpus(FILE* fp)
{
    char word[MAX_WORD_LEN];
    long long idx, new_idx;

    if (fp == NULL) { printf("Error!! No file to open"); exit(1); }

    //Index 추가 시작
    while (1)
    {
        //read word split " "
        if (feof(fp)) break;
        ReadWord(word, fp);
        train_words++;
        //check idx in vocab
        idx = word2idx(word);
        if (idx == -1)
        {
            new_idx = Addword2vocab(word);
            subword_corpus(word, new_idx);
        }
        else vocab[idx].freq++;

        if (vocab_size > vocab_hash_size * 0.7) continue;
    }
    printf("train_words ::  %lld  |  char_size ::  %lld  \n", train_words, ch_size);

}

void Make_Large_Corpus(char file_path[][100])
{
    for (long long i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;
    for (long long i = 0; i < ch_hash_size; i++) ch_hash[i] = -1;
    vocab_size = 0;
    ch_size = 0;
    Addword2vocab((char*)"</s>");

    for (int path = 0; path < num; path++)
    {
        FILE* fp;
        printf("%s\n", file_path[path]);
        fp = fopen(file_path[path], "r");

        Make_corpus(fp);
        fclose(fp);

        if (vocab_size >= vocab_hash_size * 0.7)  Reducevocab();

    }
    SortVocab();
}

//train file path
void GetfileList(char file_path[][100], char* path)
{
    long h_file;
    char search_Path[100];


    _finddata_t file_search;
    int i = 0;
    sprintf_s(search_Path, "%s/*.*", path);
    if ((h_file = _findfirst(search_Path, &file_search)) == -1L)
    {
        printf("No files in current directory!\n");
    }
    else {
        do {
            if (file_search.name[0] != '.')
            {
                strcpy(file_path[i], path);
                strcat(file_path[i], file_search.name);
                i++;
            }
        } while (_findnext(h_file, &file_search) == 0);
        _findclose(h_file);
    }
    printf("File_Detecting Finished\n");
}

void Init_Net()
{
    //자료형과 비트연산을 이용한 랜덤 생성: rand 함수 안쓰고
    //0xFFFF = 65536
    //-0.5 ~ 0.5 / embed_weight (normalize)
    unsigned long long random = 1;
    //HS부터
    //Weight 초기화

    //Initialize Weight
    for (int i = 0; i < vocab_size; i++)
    {
        for (int j = 0; j < embed_size; j++)
        {
            random = random * (unsigned long long)25214903917 + 11;
            Weight_emb[i * embed_size + j] = (float)(((random & 0xFFFF) / float(65536)) - (float)0.5) / embed_size;
        }
    }

    if (negative_sampling > 0)  for (int i = 0; i < vocab_size; i++) for (int j = 0; j < embed_size; j++)
    {
        NS_Weight[i * embed_size + j] = 0;
    }

    if (n_gram > 0) for (int i = 0; i < ch_size; i++) for (int j = 0; j < embed_size; j++)
    {
        random = random * (unsigned long long)25214903917 + 11;
        ch_Weight[i * embed_size + j] = (float)(((random & 0xFFFF) / float(65536)) - (float)0.5) / embed_size;
    }

}

void* Trainthread(int id)
{
    unsigned long long sentence[MAX_SEN + 1], rand_gen = 1, iteration, train_word_count = 0, l1, l2, idx, real_train_words = 0, target;
    int sen_pos = 0, sen_len = 0, word, target_pos, train_word;
    //1 단어씩 gradient descent 할 거 이므로 1개 벡터만 grad 필요
    float* hidden = (float*)calloc(embed_size, sizeof(float));
    float* dh = (float*)calloc(embed_size, sizeof(float));
    //float *HS_grad = (float *)calloc(embed_size, sizeof(float));
    float f, g, loss, alpha;
    int pie, window, label;
    cout << id << endl;
    pie = num / num_thread;
    if (num % num_thread != 0)
    {
        cout << "Please check thread number" << endl;
        exit(1);
    }

    clock_t now;

    //Train file list
    start = clock();
    iteration = 0;
    for (int i = 0; i < epoch; i++) for (int path = (id - 1) * pie; path < id * pie; path++)
    {
        FILE* fp;
        printf("%d file start training\n", path);
        fp = fopen(file_path[path], "r");
        /*
        fseek(fp, 0, SEEK_END);
        long temp = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        cout << temp << endl;
        */
        while (1)
        {
            iteration++;
            //파일 끝에 도달하면 종료 후 다음 파일로
            if (feof(fp)) {
                break;
            }
            if (lr_decay)
            {
                alpha = lr * (1 - real_train_words / (float)(iteration * train_words + 1));
                if (alpha < lr * 0.0001) alpha = lr * 0.0001;
            }
            else alpha = lr;

            //Sentence(array) 만들기
            if (sen_len == 0) while (1)
            {
                train_word_count++;
                if (feof(fp)) break;
                //이미 해쉬 Table 써서 찾아온 index
                word = ReadWordIndex(fp);
                //모르는 단어는("UNK" 넘겨버리기)
                if (word == -1) continue;
                //문장이 끝나면 break
                if (word == 0) break;
                //파일이 끝나도 break


                //subsampling (1 - root(1e-5 / freq)) 의 확률로 해당 word 제외 = root(sample/freq_p) 확률로 선출
                if (sub_sampling > 0)
                {
                    float prob;
                    prob = (float)sqrt((train_words * sub_sampling / vocab[word].freq));
                    rand_gen = rand_gen * (unsigned long long)25214903917 + 11;
                    if (prob < (rand_gen & 0xFFFF) / float(65536)) continue;
                }
                sentence[sen_len] = word;
                sen_len++;

                if (sen_len > MAX_SEN) break;
            }

            //target word
            word = sentence[sen_pos];
            if (word == -1) continue;

            if (win_var)
            {
                rand_gen = rand_gen * (unsigned long long)25214903917 + 11;
                window = rand_gen % skip + 1;
            }
            else window = skip;

            //starting skip-gram
            if (window) for (int i = 0; i < 2 * window + 1; i++) if (i != window)
            {
                target_pos = sen_pos - window + i;

                if (target_pos < 0) continue;
                if (target_pos >= sen_len) continue;


                //train word  의 인덱스
                train_word = sentence[target_pos];
                if (train_word == -1) continue;
                for (int layer = 0; layer < embed_size; layer++) { hidden[layer] = 0; dh[layer] = 0; }
                //hidden layer(gradient) 초기화
                loss = 0;
                //cout << (int)vocab[word].len << endl;
                for (int l = 0; l < vocab[train_word].len; l++)
                {
                    int idx;
                    idx = vocab[train_word].sub_word[l];
                    if (idx == -1) break;
                    for (int layer = 0; layer < embed_size; layer++) hidden[layer] += ch_Weight[embed_size * idx + layer];
                }

                //Starting Negative_sampling
                if (negative_sampling > 0) for (int n = 0; n < negative_sampling + 1; n++)
                {
                    if (n == 0) { target = word; label = 1; }
                    else
                    {
                        rand_gen = rand_gen * (unsigned long long)25214903917 + 11;
                        target = uni_table[(rand_gen >> 16) % table_size];
                        label = 0;
                    }
                    //get subwords sum


                    l2 = target * embed_size;
                    f = 0;
                    for (int layer = 0; layer < embed_size; layer++) f += hidden[layer] * NS_Weight[l2 + layer];
                    if (f >= MAX_EXP || f <= -MAX_EXP)
                    {
                        if (grad_clip) { f *= (float)MAX_EXP / abs(f); f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; }
                        else continue;
                    }
                    else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    real_train_words++;
                    loss += f;
                    g = (label - f) * alpha;

                    //printf("%lf\n",g);
                    //backpropagate
                    for (int layer = 0; layer < embed_size; layer++) dh[layer] += g * NS_Weight[l2 + layer];
                    for (int layer = 0; layer < embed_size; layer++) NS_Weight[l2 + layer] += g * hidden[layer];
                }
                
                for (int l = 0; l < vocab[train_word].len; l++) for (int layer = 0; layer < embed_size; layer++)
                {
                    ch_Weight[vocab[train_word].sub_word[l] * embed_size + layer] += dh[layer];
                }

            }

            sen_pos++;

            if (verbose) if (iteration % 300000 == 0)
            {

                now = clock();
                float total_time = (float)(now - start + 1) * train_words / train_word_count / (float)CLOCKS_PER_SEC / 3600 / num_thread, cur_time = (float)(now - start + 1) / 1000;
                cout << "thread id  =  " << id << "|  iteration  =  " << iteration << "|  current loss  =  " << loss << "|  time spending  =  " << cur_time << "|  Real trained word =  " << train_word_count << endl;
                cout << "expect time for end =  " << total_time << "|  total words =   " << train_words / num_thread << "|   Time Left  =  " << total_time - cur_time / (float)3600 << endl;
                //cout << g << "         " <<  f <<  "       " << Weight_emb[1000 * embed_size + 100] << endl;

                //long temp = ftell(fp);
                //cout << temp << endl;
                //printf("\niteration : %ld   |  current loss = %f | time spending = %f  | Real trained words = %lld \n", iteration, loss, (float)( now - start + 1), train_word_count);
            }
            if (sen_pos >= sen_len)
            {
                sen_len = 0;
                //word position for training
                sen_pos = 0;
                continue;
            }

        }

        fclose(fp);
    }
    _endthreadex(0);
}

unsigned int WINAPI TrainModelThread_win(void* tid) {
    int* p = (int*)tid;
    Trainthread(*p);
    return 0;
}

int val(const void* a, const void* b)
{
    // -> 포인터 . value
    float* test1 = (float*)a;
    float* test2 = (float*)b;

    if (test1 < test2) return 1;
    if (test1 > test2) return -1;
    return 0;
}

int* argmax(float* a, int size, int top = 5)
{
    float max;
    int* arg = (int*)calloc(top, sizeof(int));

    for (int j = 0; j < top; j++)
    {
        max = a[0];
        int temp = 0;
        for (int i = 0; i < size; i++)
        {
            if (a[i] > max) { max = a[i]; temp = i; }
            else continue;
        }
        //cout << max << endl;
        arg[j] = temp;
        a[temp] = 0;
    }
    return arg;
}

int* cos_similarity(float* word_vec, int top = 5)
{
    float* similarity;
    int i, layer;
    float norm = 0;

    for (layer = 0; layer < embed_size; layer++) norm += word_vec[layer] * word_vec[layer];
    norm = sqrt(norm);
    for (layer = 0; layer < embed_size; layer++) word_vec[layer] /= norm;


    similarity = (float*)calloc(vocab_size, sizeof(float));
    for (i = 0; i < vocab_size; i++) for (layer = 0; layer < embed_size; layer++) similarity[i] += Weight_emb[i * embed_size + layer] * word_vec[layer];
    int* arg = argmax(similarity, vocab_size);
    free(similarity);
    return arg;
    //for (i = 0; i < top; i++) candidate[i] = arg[i];
}

void Save_vocab()
{
    long long i;
    FILE* fo = fopen("vocab.txt", "wb");
    fprintf(fo, "%lld", train_words);
    for (i = 0; i < vocab_size; i++)
    {
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].freq);
    }
    fclose(fo);

    fo = fopen("sub_vocab.txt", "wb");
    fprintf(fo, "%lld", ch_size);
    for (i = 0; i < ch_size; i++) fprintf(fo, "%s\n", sub[i].word);
    fclose(fo);
}

void loadvocab()
{
    long long index;
    char c;
    char word[MAX_WORD_LEN];
    FILE* fin = fopen("vocab.txt", "rb");
    if (fin == NULL) exit(1);
    fscanf(fin, "%lld", &train_words);
    for (index = 0; index < vocab_hash_size; index++) vocab_hash[index] = -1;
    for (index = 0; index < ch_hash_size; index++) ch_hash[index] = -1;
    vocab_size = 0;
    ch_size = 0;
    while (1) {
        ReadWord(word, fin);
        //weight를 다시 tranfer learning 안할거면 그냥 써도 됨 but index를 유지할거면 freq등으로 정렬 후 사용
        if (feof(fin)) break;
        index = Addword2vocab(word);
        subword_corpus(word, index);
        fscanf(fin, "%lld%c", &vocab[index].freq, &c);
        //if (vocab_size % 1000 == 0) cout  << "Loading Vocab , current vocab =  " << vocab_size << endl;
    }
    //for (int i = 0; i < vocab_size; i++) train_words += vocab[i].freq;
    SortVocab();

}

void Word_score(int* score, int top = 5)
{
    int lines, sen_len = 0, word, layer, i, sort = 0;
    int sentence[4];
    float* word_vec;
    float* norm;

    score[0] = 0; score[1] = 0;
    norm = (float*)calloc(vocab_size, sizeof(float));
    word_vec = (float*)calloc(embed_size, sizeof(float));

    for (i = 0; i < vocab_size; i++)
    {
        for (layer = 0; layer < embed_size; layer++) norm[i] += Weight_emb[i * embed_size + layer] * Weight_emb[i * embed_size + layer];
        norm[i] = sqrt(norm[i]);
    }
    for (i = 0; i < vocab_size; i++) for (layer = 0; layer < embed_size; layer++) Weight_emb[i * embed_size + layer] /= norm[i];

    FILE* fp;
    fp = fopen("./questions-words.txt", "r");
    cout << vocab[0].word << endl;
    lines = 0;
    while (1)
    {
        sen_len = 0;
        lines++;
        if (lines > 8869 * 2) sort = 1;
        if (verbose) if (lines % 1000 == 0) {
            int bar = (int)(30 * (float)lines / (39000));
            system("cls");
            for (int i = 0; i < bar; i++) cout << "=";
            cout << endl;
            cout << "Scoring " << lines << "th complete" << endl; cout << (float)lines / (float)(39000) * 100 << " % complete" << endl;
        }
        if (feof(fp))
        {
            break;
        }
        //Sentence(array) 만들기
        if (sen_len == 0) while (1)
        {
            if (feof(fp)) break;
            //이미 해쉬 Table 써서 찾아온 index
            word = ReadWordIndex(fp);
            //if (strcmp(vocab[word].word, ":") == 0) { cout << 1 << endl;  break; }
            //문장이 끝나면 break
            if (word == -1) continue;
            if (word == 0) break;
            //파일이 끝나도 break
            //if (!strcmp(vocab[word].word, ":")) break;

            sentence[sen_len] = word;
            sen_len++;
            if (sen_len >= 4) break;
        }

        //for (int i = 0; i < sen_len; i++) cout << vocab[sentence[i]].word << "  " << sentence[i] << "  ";
        //cout << sen_len << endl;
        if (sen_len != 4) continue;
        if (sentence[0] == -1 || sentence[1] == -1 || sentence[2] == -1 || sentence[3] == -1) continue;

        for (layer = 0; layer < embed_size; layer++) word_vec[layer] = 0;
        //sentence = [w1, w2, w3 ,w4]
        //w2 + w3 - w1 == w4?
        for (i = 1; i < 3; i++) for (layer = 0; layer < embed_size; layer++) word_vec[layer] += Weight_emb[sentence[i] * embed_size + layer];
        for (layer = 0; layer < embed_size; layer++) word_vec[layer] -= Weight_emb[sentence[0] * embed_size + layer];

        int* candidate = cos_similarity(word_vec, top);
        for (i = 0; i < top; i++) {
            //if (lines % 100 == 0) cout << "target word  " << vocab[sentence[3]].word << "  candidate word  " <<   vocab[candidate[i]].word <<  "    "   << endl;
            if (sentence[3] == candidate[i]) { score[sort]++; break; }
        }
        free(candidate);

    }

    fclose(fp);
    free(word_vec);
    free(norm);

}

void train()
{
    int* cap;
    cap = (int*)malloc(sizeof(int) * num_thread);
    for (int i = 0; i < num_thread; i++) cap[i] = i + 1;
    HANDLE* pt = (HANDLE*)malloc(num_thread * sizeof(HANDLE));
    for (int i = 0; i < num_thread; i++) {
        pt[i] = (HANDLE)_beginthreadex(NULL, 0, TrainModelThread_win, &cap[i], 0, NULL);
    }
    WaitForMultipleObjects(num_thread, pt, TRUE, INFINITE);
    for (int i = 0; i < num_thread; i++) {
        CloseHandle(pt[i]);
    }

    free(pt);
    free(cap);

}

//Vocab Weight from Subvocab
void create_vocab_weight()
{
    float* vocab_vec;
    vocab_vec = (float*)calloc(embed_size, sizeof(float));

    for (int i = 0; i < vocab_size; i++)
    {
        for (int j = 0; j < vocab[i].len; j++)
        {
            for (int k = 0; k < embed_size; k++) {
                vocab_vec[k] += ch_Weight[vocab[i].sub_word[j] * embed_size + k];
            }
        }
        for (int j = 0; j < embed_size; j++) Weight_emb[i * embed_size + j] = vocab_vec[j];
    }
}


int main()
{
    int temp;
    int score[2];
    char path[100] = "./1-billion-word/training-monolingual.tokenized.shuffled/";
    char save_path[100] = "./weight_sub.txt";
    expTable = (float*)malloc(sizeof(float) * (EXP_TABLE_SIZE + 1));
    for (int i = 0; i < EXP_TABLE_SIZE; i++)
    {
        //-MAX_EXP ~ MAX_EXP 까지 resolution개
        expTable[i] = exp((i / float(EXP_TABLE_SIZE) * 2 - 1) * MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }

    GetfileList(file_path, path);
    vocab_hash = (int*)malloc(sizeof(long) * vocab_hash_size);
    vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));

    if (n_gram > 0)
    {
        ch_hash = (int*)malloc(sizeof(long) * ch_hash_size);
        sub = (struct char_word*)calloc(ch_max_size, sizeof(struct char_word));
    }

    //Make_Large_Corpus(file_path);
    //Save_vocab();
    loadvocab();
    cout << train_words << "  ch  " << ch_size << " ch_max " << ch_max_size << endl;
    cout << vocab_size << endl;
    //Initialize weight
    Weight_emb = (float*)_aligned_malloc((long long)vocab_size * embed_size * sizeof(float), 128);
    NS_Weight = (float*)_aligned_malloc((long long)vocab_size * embed_size * sizeof(float), 128);
    cout << (long long)ch_size * embed_size * sizeof(float) << endl;
    ch_Weight = (float*)_aligned_malloc((long long)ch_size * embed_size * sizeof(float), 128);

    Init_Net();
    //load();
    unigram_table();

    train();
    create_vocab_weight();
    Word_score(score);
    for (int i = 0; i < 2; i++) cout << score[i] << endl;
    FILE* fo = fopen("score_subword.txt", "ab");
    fseek(fo, 0, SEEK_END);
    fprintf(fo, "Devide by Window_size | NS-skip-gram | grad cliff %d | lr_decay %d | window_random = %d | skip_size = 5 | lr = %f | subsampling = %f | min_freq = %d | sem = %d  | syn = %d", grad_clip, lr_decay, win_var, lr, sub_sampling, min_freq, score[0], score[1]);
    fprintf(fo, "\n");

    fclose(fo);

    //free(Weight_emb);
    //free(HS_Weight);


}