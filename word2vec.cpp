#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <io.h>

struct vocab_word {
   long long freq;
   int* point;
   char *word, *direction_path, path_len;
};

const int vocab_hash_size = 30000000;
#define MAX_SEN 1000
#define MAX_WORD_LEN 100
#define MAX_CODE_LEN 40

char train_file[MAX_SEN];
int skip = 5;
long long vocab_size = 0, vocab_max_size = 1000;
struct vocab_word* vocab;
int* vocab_hash;

void ReadWord(char* word, FILE* fp)
{
   //open 된 fp 와 buffer 수만큼의 word --> 하나의 word 추출
   //" ", \n, \t 제외
   // NULL 값도 포함되어 있음 --> Word에
   int ch;
   int i = 0;
   while (!feof(fp))
   {
      ch = fgetc(fp);
      if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
      {
         break;
      }

      word[i] = ch;
      i++;
      if (i > MAX_WORD_LEN) i--;

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
   unsigned int hash = GetHash(word);

   char length = strlen(word) + 1;

   if (length > MAX_WORD_LEN) length = MAX_WORD_LEN;
   //메모리가 해제되었는데 접근한 경우 segmentation fault
   vocab[vocab_size].word = (char*)calloc(length, sizeof(char));

   strcpy(vocab[vocab_size].word, word);

   vocab->freq = 1;
   if (vocab_size + 3 > vocab_max_size)
   {
      vocab_max_size += 1000;
      vocab = (struct vocab_word*)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
   }

   while (1)
   {
      if (vocab_hash[hash] == -1) break;
      hash = (hash + 1) % vocab_hash_size;
   }

   vocab_hash[hash] = vocab_size;

   vocab_size++;

}

int f(const void* a, const void* b)
{
   // -> 포인터 . value
   struct vocab_word * test1 = (struct vocab_word*)a;
   struct vocab_word* test2 = (struct vocab_word*)b;

   if (test1->freq < test2 -> freq) return 1;
   if (test1 -> freq > test2 -> freq) return -1;
   return 0;
}

void SortVocab()
{
   long long i;
   long long min_freq = 3;
   unsigned int hash;
   //Huffman 사용하기 전 Vocab Sort 함수 & freq <min 이하는 제거
   //"UNK" 빼고 나머지 내림차순
   qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), f);
   for (i = 0; i < vocab_hash_size; i++) vocab_hash[i] = -1;

   long long size = vocab_size;
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
         vocab_hash[hash] = i;
      }
   }
   //앞에서부터 짤림 --> 내림차순이라 줄어든 vocab_size 개수만큼 짜르면 알아서 cutting
   vocab = (struct vocab_word*)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

   for (i = 0; i < vocab_size; i++)
   {
      vocab[i].direction_path = (char*)calloc(MAX_CODE_LEN, sizeof(char));
      vocab[i].point = (int*)calloc(MAX_CODE_LEN, sizeof(int));
   }

   printf("final vocab size :: %lld", vocab_size);

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
   for (i = vocab_size; i < 2 * vocab_size ; i++) count[i] = 1e10;
   
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
   long long train_words = 0;
   long long idx;
   
   if (fp == NULL) { printf("Error!! No file to open"); exit(1); }

   //Index 추가 시작
   while (1)
   {
      //read word split " "
      if (feof(fp)) break;
      ReadWord(word, fp);
      //check idx in vocab
      idx = word2idx(word);
      if (idx == -1)
      {
         Addword2vocab(word);
         train_words++;
      }
      else vocab[idx].freq++;
   
      if (vocab_size > vocab_hash_size * 0.7) continue;
   
   }
   printf("train_words ::  %lld", train_words);
   
}

void Make_Large_Corpus(char file_path[][100])
{
   for (long long i = 0; i < vocab_hash_size; i++)
   {
      vocab_hash[i] = -1;
   }
   Addword2vocab((char*)"UNK");
   vocab_size = 0;
   for (int path = 0; path < 2; path++)
   {
      FILE* fp;
      printf("%s\n", file_path[path]);
      fp = fopen(file_path[path], "r");

      Make_corpus(fp);
      fclose(fp); 
   }
   SortVocab();
   Huffman();
}



//train file path
void GetfileList(char file_path[][100], char *path)
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


int main()
{
   char file_path[100][100];
   char path[100] = "./1-billion-word/training-monolingual.tokenized.shuffled/";

   GetfileList(file_path, path);

   vocab_hash = (int*)malloc(sizeof(long) * vocab_hash_size);
   vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));

   Make_Large_Corpus(file_path);

   for (int i = 0 ; i < 10; i++)
   {
      printf("%s\n",vocab[i].point);
   }
   //Huffman();

}
