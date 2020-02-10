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

struct vocab_word {
    long long freq;
    int* point;
    char* word, * direction_path, path_len;
};

const int vocab_hash_size = 3000000;
#define MAX_SEN 1000
#define MAX_WORD_LEN 100
#define MAX_CODE_LEN 40
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

int min_freq = 3;
int skip = 5, negative_sampling = 5, embed_size = 300;
long long vocab_size = 0, vocab_max_size = 1000, train_words = 0;
struct vocab_word* vocab;
int* vocab_hash;
clock_t start;
int num_thread = 3;
char file_path[100][100];
int num = 33;
int epoch = 1;
float lr = 0.0025;
float sub_sampling = 0.00001;

float* Weight_emb, * HS_Weight, * Nega_emb, * expTable;

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
            /*
            if (i > 0)
            {
               if (ch == '\n') ungetc(ch,fp);
               break;
            }
            */
            if (ch == '\n')
            {
                //문장 구분을 위한 추가
                strcpy(word, (char*)"</s>");
                return;
            }
            break;
        }
        word[i] = ch;
        i++;
        if (i >= MAX_WORD_LEN - 1) i--;

    }
    // 이거 왜 넣는지 이해 x
    word[i] = 0;
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

void Addword2vocab(char* word)
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

}

int f(const void* a, const void* b)
{
    // -> 포인터 . value
    struct vocab_word* test1 = (struct vocab_word*)a;
    struct vocab_word* test2 = (struct vocab_word*)b;

    if (test1->freq < test2->freq) return 1;
    if (test1->freq > test2->freq) return -1;
    return 0;
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
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), f);
    for (i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

    size = vocab_size;
    for (i = 0; i < size; i++)
    {
        //printf("%d   ", freq);
        if ((vocab[i].freq < min_freq) && (i != 0))
        {
            vocab_size--;
            free(vocab[i].word);
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

    for (i = 0; i < vocab_size; i++)
    {
        vocab[i].direction_path = (char*)calloc(MAX_CODE_LEN, sizeof(char));
        vocab[i].point = (int*)calloc(MAX_CODE_LEN, sizeof(int));
    }

    printf("final vocab size :: %lld \n", vocab_size);

}


void Huffman()
{
    //사용 전 vocab 정렬(freq 높은 기준으로)
    long long i, pos1, pos2, min1, min2, a;
    int length;
    char code[MAX_CODE_LEN];
    long long point[MAX_CODE_LEN];
    //이거 왜 2V + 1?
    long long* count = (long long*)calloc(2 * vocab_size + 1, sizeof(long long));
    long long* binary = (long long*)calloc(2 * vocab_size + 1, sizeof(long long));
    long long* parent_node = (long long*)calloc(2 * vocab_size + 1, sizeof(long long));
    for (i = 0; i < vocab_size; i++) count[i] = vocab[i].freq;
    for (i = vocab_size; i < 2 * vocab_size; i++) count[i] = 1e10;

    pos1 = vocab_size - 1;
    pos2 = vocab_size;

    for (i = 0; i < vocab_size - 1; i++)
    {

        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2])
            {
                min1 = pos1;
                pos1--;
            }
            else
            {
                min1 = pos2;
                pos2++;
            }
        }
        else
        {
            min1 = pos2;
            pos2++;
        }

        if (pos1 >= 0)
        {
            if (count[pos1] < count[pos2])
            {
                min2 = pos1;
                pos1--;
            }
            else
            {
                min2 = pos2;
                pos2++;
            }
        }
        else
        {
            min2 = pos2;
            pos2++;
        }

        count[vocab_size + i] = count[min1] + count[min2];
        parent_node[min1] = vocab_size + i;
        parent_node[min2] = vocab_size + i;
        //min2  > min1 : 큰놈이 1로 갈 수 있도록
        binary[min2] = 1;
    }
    for (i = 0; i < vocab_size; i++)
    {
        a = i;
        length = 0;
        while (1)
        {
            code[length] = binary[a];
            point[length] = a;
            length++;
            a = parent_node[a];
            //어자피 a 자체가 parent 노드 값
            if (a == vocab_size * 2 - 2) break;
        }

        vocab[i].path_len = length;
        vocab[i].point[0] = vocab_size - 2;

        for (int b = 0; b < length; b++)
        {
            vocab[i].direction_path[length - b - 1] = code[b];
            vocab[i].point[length - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);

}


//vocab에 freq, path, word, id들 계산 --> path = Huffman // word == ReadWordID, id // freq
void Make_corpus(FILE* fp)
{
    char word[MAX_WORD_LEN];
    long long idx;

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
            Addword2vocab(word);
        }
        else vocab[idx].freq++;

        if (vocab_size > vocab_hash_size * 0.7) continue;

    }
    printf("train_words ::  %lld \n", train_words);

}

void Make_Large_Corpus(char file_path[][100])
{
    for (long long i = 0; i < vocab_hash_size; i++)
    {
        vocab_hash[i] = -1;
    }
    Addword2vocab((char*)"</s>");
    vocab_size = 0;
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
    Huffman();
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

    Weight_emb = (float*)_aligned_malloc((long long)vocab_size * embed_size * sizeof(float), 128);
    HS_Weight = (float*)_aligned_malloc((long long)vocab_size * embed_size * sizeof(float), 128);

    //Initialize Weight
    for (int i = 0; i < vocab_size; i++)
    {
        for (int j = 0; j < embed_size; j++)
        {
            random = random * (unsigned long long)25214903917 + 11;
            Weight_emb[i * embed_size + j] = (float)(((random & 0xFFFF) / float(65536)) - (float)0.5) / embed_size;
        }
    }
    for (int i = 0; i < vocab_size; i++)
    {
        for (int j = 0; j < embed_size; j++)
        {
            random = random * (unsigned long long )25214903917 + 11;
            HS_Weight[i * embed_size + j] = (float)(((random & 0xFFFF) / float(65536)) - (float)0.5) / embed_size;
        }
    }
}

void* Trainthread(int id)
{
    unsigned long long sentence[MAX_SEN + 1], rand_gen = 1, iteration, train_word_count = 0;
    int sen_pos = 0, sen_len = 0, word, target_pos, train_word;
    //1 단어씩 gradient descent 할 거 이므로 1개 벡터만 grad 필요
    float* hidden = (float*)calloc(embed_size, sizeof(float));
    //float *HS_grad = (float *)calloc(embed_size, sizeof(float));
    float f, g, loss;
    int pie;
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
                cout << "Hello world" << endl;
                break;
            }
            //Sentence(array) 만들기
            if (sen_len == 0) while (1)
            {
                train_word_count++;
                if (feof(fp)) break;
                //이미 해쉬 Table 써서 찾아온 index
                word = ReadWordIndex(fp);
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
            for (int i = 0; i < 2 * skip + 1; i++) if (i != skip)
            {
                target_pos = sen_pos - skip + i;

                if (target_pos < 0) continue;
                if (target_pos >= sen_len) continue;

                //train word  의 인덱스
                train_word = sentence[target_pos];

                //hidden layer(gradient) 초기화
                for (int layer = 0; layer < embed_size; layer++) hidden[layer] = 0;
                loss = 0;
                for (int j = 0; j < vocab[word].path_len; j++)
                {
                    int idx = vocab[word].point[j];
                    f = 0;
                    //feed forward
                    for (int layer = 0; layer < embed_size; layer++) f += Weight_emb[train_word * embed_size + layer] * HS_Weight[idx * embed_size + layer];
                    //sigmoid  --> Table로 변환하면 가속화
                    //f = (float)1 / (float)(1 + exp(-f));
                    if (f >= MAX_EXP || f <= -MAX_EXP) continue;
                    else f = expTable[(int)(f / 2 / MAX_EXP * EXP_TABLE_SIZE + EXP_TABLE_SIZE / 2)];

                    loss += f;
                    //gradient
                    //(y - t) 이 원래 dloss 인데 여기서는 huffman tree 에서 방향을 바꿔서 - 를 부여
                    //(f - idx)로 학습 해보기
                    g = (1 - f - (float)vocab[word].direction_path[j]) * lr;

                    //printf("%lf\n",g);
                    //backpropagate
                    for (int layer = 0; layer < embed_size; layer++) hidden[layer] += g * HS_Weight[word * embed_size + layer];
                    for (int layer = 0; layer < embed_size; layer++) HS_Weight[word * embed_size + layer] += g * Weight_emb[train_word * embed_size + layer];
                }

                for (int layer = 0; layer < embed_size; layer++)
                {
                    Weight_emb[train_word * embed_size + layer] += hidden[layer];
                }

            }

            sen_pos++;

            if (iteration % 300000 == 0)
            {

                now = clock();
                float total_time = (float)(now - start + 1) * train_words / train_word_count / (float)CLOCKS_PER_SEC / 3600 / num_thread, cur_time = (float)(now - start + 1) / 1000;
                cout << "thread id  =  " << id << "|  iteration  =  " << iteration << "|  current loss  =  " << loss << "|  time spending  =  " << cur_time << "|  Real trained word =  " << train_word_count << endl;
                cout << "expect time for end =  " << total_time << "|  total words =   " << train_words / num_thread << "|   Time Left  =  " << total_time - cur_time / (float)3600 << endl;
                cout << g << "         " <<  f <<  "       " << Weight_emb[1000 * embed_size + 100] << endl;

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

int *argmax(float* a, int size, int top = 5)
{
    float max;
    int *arg = (int*)calloc(top, sizeof(int));

    for (int j = 0; j < top; j++)
    {
        max = a[0];
        int temp = 0;
        for (int i = 0; i < size; i++)
        {
            if (a[i] > max) { max = a[i]; temp = i; }
            else continue;
        }

        arg[j] = temp;
        a[temp] = 0;
    }
    return arg;
}
void cos_similarity(float* word_vec, float* candidate, int top = 5)
{
    float* similarity;
    int i, layer;
    float norm = 0;

    for (layer = 0; layer < embed_size; layer++) norm += word_vec[layer] * word_vec[layer];
    norm = sqrt(norm);
    for (layer = 0; layer < embed_size; layer++) word_vec[layer] /= norm;

    
    similarity = (float*)calloc(sizeof(float), vocab_size);
    for (i = 0; i < vocab_size; i++) for (layer = 0; layer < embed_size; layer++) similarity[i] += Weight_emb[i * embed_size + layer] * word_vec[layer];
    int* arg = argmax(similarity, vocab_size);

    for (i = 0; i < top; i++) candidate[i] = arg[i];

}

void Word_score(int* score, int top = 5)
{
    int lines, sen_len = 0, word, layer, i, sort = 0;
    int sentence[4];
    float* word_vec;
    float* norm;
    float* candidate;

    score[0] = 0; score[1] = 0;
    norm = (float*)calloc(sizeof(float), vocab_size);
    word_vec = (float*)calloc(sizeof(float), embed_size);
    candidate = (float*)calloc(sizeof(float), top);

    for (i = 0; i < vocab_size; i++)
    {
        for (layer = 0; layer < embed_size; layer++) norm[i] += Weight_emb[i * embed_size + layer] * Weight_emb[i * embed_size + layer];
        norm[i] = sqrt(norm[i]);
    }
    for (i = 0; i < vocab_size; i++) for (layer = 0; layer < embed_size; layer++) Weight_emb[i * embed_size + layer] /= norm[i];

    FILE* fp;
    fp = fopen("./questions-words.txt", "r");

    lines = 0;
    while (1)
    {
        sen_len = 0;
        lines++;
        if (lines > 8875) sort = 1;
        if (lines % 1000 == 0) cout << "Scoring " << lines << "th complete" << endl;
        if (feof(fp))
        {
            cout << "Hello world" << endl;
            break;
        }
        //Sentence(array) 만들기
        if (sen_len == 0) while (1)
        {
            if (feof(fp)) break;
            //이미 해쉬 Table 써서 찾아온 index
            word = ReadWordIndex(fp);
            //문장이 끝나면 break
            if (word == -1) continue;
            if (word == 0) break;
            //파일이 끝나도 break
            //if (!strcmp(vocab[word].word, ":")) break;

            sentence[sen_len] = word;
            sen_len++;
            if (sen_len > 4) break;
        }

        if (sen_len != 4) continue;
        if (strcmp(vocab[sentence[0]].word, ":") == 0) continue;

        for (layer = 0; layer < embed_size; layer++) word_vec[layer] = 0;
        //sentence = [w1, w2, w3 ,w4]
        //w2 + w3 - w1 == w4?
        for (i = 1; i < 3; i++) for (layer = 0; layer < embed_size; layer++) word_vec[layer] += Weight_emb[sentence[i] * embed_size + layer];
        for (layer = 0; layer < embed_size; layer++) word_vec[layer] -= Weight_emb[sentence[0] * embed_size + layer];

        cos_similarity(word_vec, candidate, top);
        for (i = 0; i < top; i++) if (sentence[3] == candidate[i]) { score[sort]++; break; }

    }

    fclose(fp);
    free(word_vec);
    free(norm);
    free(candidate);

}

void save_all()
{
    FILE* fp;
    fopen_s(&fp, "./weight.txt", "wb");

    if (fp)
    {

        fwrite(&vocab_size, sizeof(long long), 1, fp);
        fwrite(Weight_emb, sizeof(float) * vocab_size * embed_size, 1, fp);
        fclose(fp);

        /*
        fprintf(fp, "%lld %lld\n", vocab_size, embed_size);
        for (int a = 0; a < vocab_size; a++) {
            fprintf(fp, "%s ", vocab[a].word);
            for (int b = 0; b < embed_size; b++) fwrite(&Weight_emb[a * embed_size + b], sizeof(float), 1, fo);
            for (int b = 0; b < embed_size; b++) fprintf(fo, "%lf ", Weight_emb[a * embed_size + b]);
            fprintf(fp, "\n");
        }
        */
    }

    else
    {
        printf("데이터 저장 실패\n");
    }
}
void load()
{
    FILE* fs;
    float* px;

    if (fopen_s(&fs, "./weight.txt", "rb") != 0)
    {
        printf("Cannot Open file \n");
        exit(1);
    }

    //px = (float*)_aligned_malloc(sizeof(float) * vocab_size * embed_size, 128);
    fread(&vocab_size, sizeof(long long), 1, fs);
    Weight_emb = (float*)_aligned_malloc(sizeof(float) * vocab_size * embed_size, 128);
    //fread(&vocab_hash_size , sizeof(int) , 1, fs);
    //for (int i = 0; i < vocab_size; i++) for (int layer = 0; layer < embed_size; layer++) 
    fread(Weight_emb, sizeof(float) * vocab_size * embed_size, 1, fs);
    //fread(vocab, sizeof(struct vocab_word) * vocab_size, 1, fs);
    //fread(vocab_hash, sizeof(int) * vocab_hash_size, 1, fs);
    fclose(fs);
}



int main()
{
    int temp;
    int score[2];
    char path[100] = "./1-billion-word/training-monolingual.tokenized.shuffled/";
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
    Make_Large_Corpus(file_path);
    //Initialize weight
    Init_Net();
    //save_all();
    //load();
    //for (int i = 10000; i < 10003; i++) for (int j = 0; j < 15; j++) cout << Weight_emb[i * embed_size + j] << endl;
    
    
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
    save_all();
    
    //cout << "" << endl;
    //load();



    
    Word_score(score);
    for (int i = 0; i < 2; i++)
    {
        cout << score[i] << endl;
    }
    
    //Train(file_path, epoch, lr, sub_sampling);
    

    //free(Weight_emb);
    //free(HS_Weight);
}
