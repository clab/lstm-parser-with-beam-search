#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include "cnn/cfsm-builder.h"
#include "cnn/dict.h"
#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "c2.h"

using namespace cnn;

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;
unsigned OUTPUT_DIM; //the size of the output

unsigned LSTM_CHAR_OUTPUT_DIM = 100; //Miguel
bool USE_SPELLING = false;

bool USE_POS = false;

constexpr const char* ROOT_SYMBOL = "ROOT";
static constexpr const char* EPSILON = "EPSILON";

unsigned kROOT_SYMBOL = 0;
unsigned kEPSILON = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;

unsigned CHAR_SIZE = 255; //size of ascii chars... Miguel
cnn::ClassFactoredSoftmaxBuilder *cfsm = nullptr;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
        ("clusters,c", po::value<string>(), "Clusters word clusters file")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("use_spelling,S", "Use spelling model") //Miguel. Spelling model
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {


  LSTMBuilder output_lstm; //(MT) added new LSTM for the output stack.

  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_r; // relation embeddings
  LookupParameters* p_p; // pos tag embeddings
  LookupParameters* p_o; //output word embeddings
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_O; // output lstm to parser state, not sure if we need this
  Parameters* p_H; // head matrix for composition function
  Parameters* p_D; // dependency matrix for composition function
  Parameters* p_R; // relation matrix for composition function
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_p2l; // POS to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_p2o; // parser state to output word, need this if the action choosen is generate or output
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack
  Parameters* p_output_guard;  // end of output

  Parameters* p_start_of_word;//Miguel -->dummy <s> symbol
  Parameters* p_end_of_word; //Miguel --> dummy </s> symbol
  LookupParameters* char_emb; //Miguel-> mapping of characters to vectors 

  LSTMBuilder fw_char_lstm; // Miguel
  LSTMBuilder bw_char_lstm; //Miguel


  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      output_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model), //(MT) same sizes as the other lstm that contain tokens.
      buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),

      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM, 1})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM, 1})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {REL_DIM, 1})),
      p_o(model->add_lookup_parameters(OUTPUT_DIM, {LSTM_INPUT_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM, 1})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_O(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_H(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_D(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_R(model->add_parameters({LSTM_INPUT_DIM, REL_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_p2o(model->add_parameters({OUTPUT_DIM, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM, 1})),
      p_abias(model->add_parameters({ACTION_SIZE, 1})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_output_guard(model->add_parameters({LSTM_INPUT_DIM, 1})),  //this is the last element of the output

      p_start_of_word(model->add_parameters({LSTM_INPUT_DIM, 1})), //Miguel
      p_end_of_word(model->add_parameters({LSTM_INPUT_DIM, 1})), //Miguel 

      char_emb(model->add_lookup_parameters(CHAR_SIZE, {INPUT_DIM, 1})),//Miguel

//      fw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM, model), //Miguel
//      bw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM,  model), //Miguel

      fw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, model), //Miguel 
      bw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, model) /*Miguel*/ {
    if (USE_POS) {
      p_p = model->add_lookup_parameters(POS_SIZE, {POS_DIM, 1});
      p_p2l = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM, 1});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }


//(MT)  the possible actions are SHIFT, OUTPUT, SWAP, DELETE, COPY, and OUT_E (output epsilon)
static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, vector<int> stacki, unsigned action_count, unsigned eps_count, unsigned copy_count) {
  //(MT) we need this. Especially for swap and shift. Dan, this is basically some constraints in order to avoid invalid actions like SHIFT when the buffer is empty.
  bool stop = false; //this stops allowing anything other than shift, delete, and output after this point, to stop the system from going haywire

  bool no_eps = false;

  bool no_copy = false;

  if(eps_count > 1000) no_eps = true;
  if(copy_count > 1000) no_copy = true;
  if (action_count > 10000)  stop = true;
  
  if (a[1]=='W' && (ssize<3 || stop)) return true; //MIGUEL  //(MT) this is needed. We cannot swap if the stack has less than 2 elements.
  /*
  if (a[1]=='W') { //MIGUEL //(MT) this is again needed. SWAP is the same as in the parser. So nice!
                   //Dan  //This may actually not be needed-there is a lot of repeating reorderings
        int top=stacki[stacki.size()-1];
        int sec=stacki[stacki.size()-2];

        if (sec>top) return true;
  }*/

  //you can always output epsilons
  //could be a problem?
  //changed to only allow epsilons if there hasn't been too many actions performed
  if(a[0] == 'G' && (stop || no_eps)) return true;

  bool is_shift = (a[0] == 'S' && a[1]=='H');  //MIGUEL
  bool is_delete = (a[0] == 'D'); //this takes care of delete, copy, and output
  bool is_copy = (a[0] == 'C');
  bool is_output = (a[0] == 'O');


  if ((is_delete || is_shift) && bsize == 1) return true;
  if (is_copy && (bsize == 1 || stop || no_copy)) return true; //(MT) this is okay, we cannot SHIFT if the Stack is empty
  //or copy or delete, as those operations are done on the stack
  
  
  if(is_output && ssize == 1) return true;
//  if (is_reduce && ssize < 3) return true;  //(MT) this was not okay, since we are not creating arcs between two words in the stack.
  //if (bsize == 2 && // ROOT is the only thing remaining on buffer
  //    ssize > 2 && // there is more than a single element on the stack
   //   is_shift) return true; //(MT) we might need to think about this one.. Probably not needed.
  //i don't think we do? -DAN

  // only attach left to ROOT
//  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true; //(MT) this is not needed, we should change this accordingly. (same reason as above).


  return false;
}


//(MT) This is basically the decoding method. We will need to change this to make decoding with the new set of actions. SWAP and SHIFT are going to be exactly the same. The rest of the stuff needs to change.
//(MT) I change the name of the method from "compute_output" to "compute_outputs"
static map<int,int> compute_output(unsigned sent_len, const vector<unsigned>& actions, const vector<string>& setOfActions, map<int,string>* pr = nullptr) {
  map<int,int> heads;
  map<int,string> r;
  map<int,string>& rels = (pr ? *pr : r);
  for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
  vector<int> bufferi(sent_len + 1, 0), stacki(1, -999), outputi(1, -999);
  for (unsigned i = 0; i < sent_len; ++i)
    bufferi[sent_len - i] = i;
  bufferi[0] = -999;
  for (auto action: actions) { // loop over transitions for sentence
    const string& actionString=setOfActions[action];
    const char ac = actionString[0];
    const char ac2 = actionString[1];
    char ac5 = ' ';
    if(actionString.size() > 3){
      ac5 = actionString[4];
    }
    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } 
   else if (ac=='S' && ac2=='W') {
        assert(stacki.size() > 2);

//	std::cout<<"SWAP"<<"\n";
        unsigned ii = 0, jj = 0;
        jj=stacki.back();
        stacki.pop_back();

        ii=stacki.back();
        stacki.pop_back();

        bufferi.push_back(ii);

        stacki.push_back(jj);
    }
    else if (ac == 'G'){
      assert(stacki.size() > 1);
      outputi.push_back(stacki.back());
      stacki.pop_back();
    }
    else if(ac == 'D'){
      assert(bufferi.size() > 1);
      bufferi.pop_back();
    }
    else if(ac == 'C'){
      assert(bufferi.size() > 1);
      bufferi.push_back(bufferi.back());
    }
    //this is for output epsilon
    else {
      outputi.push_back(-100);
      //push -1, maybe?
    }
  }


    /*
    else { // LEFT or RIGHT //(MT) this is the part that needs to change. 

      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      unsigned depi = 0, headi = 0;
      (ac == 'R' ? depi : headi) = stacki.back();
      stacki.pop_back();
      (ac == 'R' ? headi : depi) = stacki.back();
      stacki.pop_back();
      stacki.push_back(headi);
      heads[depi] = headi;
      rels[depi] = actionString;
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  */
  return heads;
  
}

// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
//(MT) this is for character-based embeddings. Do not remove it, since it might be useful later. 
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
//(MT) this is the main thing that we need to change accordingly. Again, the decoding part is going to be similar since the core algorithm is the same or very similar.
vector<std::string> log_prob_parser(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with oovs replaced
                     const vector<unsigned>& sentPos,
                     const vector<unsigned>& correct_actions,
                     vector<string>& setOfActions,
                     const map<unsigned, std::string>& intToWords,
                     vector<std::string> correct_words,
                     double *right) {
  //for (unsigned i = 0; i < sent.size(); ++i) cerr << ' ' << intToWords.find(sent[i])->second;
  //cerr << endl;
    vector<string> results;
    const bool build_training_graph = correct_actions.size() > 0;

    output_lstm.new_graph(*hg); //(MT) new graph also for the output_lstm
    stack_lstm.new_graph(*hg);
    buffer_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    cfsm->new_graph(*hg);
    output_lstm.start_new_sequence(); //(MT) 
    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression O = parameter(*hg, p_O);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression p2o = parameter(*hg, p_p2o);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    Expression word_end = parameter(*hg, p_end_of_word); //Miguel
    Expression word_start = parameter(*hg, p_start_of_word); //Miguel
    Expression eps = lookup(*hg, p_w, kEPSILON); //expression for epsilon
    Expression eps_eps = affine_transform({ib, w2l, eps});;

    if (USE_SPELLING){
       fw_char_lstm.new_graph(*hg);
        //    fw_char_lstm.add_parameter_edges(hg);

       bw_char_lstm.new_graph(*hg);
       //    bw_char_lstm.add_parameter_edges(hg);
    }



    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      //Expression w = lookup(*hg, p_w, sent[i]);

      unsigned wi=sent[i];
      std::string ww=intToWords.at(wi);
      Expression w;
      /**********SPELLING MODEL*****************/ //(MT) all this IF can be ignored and it can be the same if we want to use character-based embeddings. For now, let's keep it like this.
      if (USE_SPELLING) {
        //std::cout<<"using spelling"<<"\n";
        if (ww.length()==4  && ww[0]=='R' && ww[1]=='O' && ww[2]=='O' && ww[3]=='T'){
          w=lookup(*hg, p_w, sent[i]); //we do not need a LSTM encoding for the root word, so we put it directly-.
        }
        else {

            fw_char_lstm.start_new_sequence();
            //cerr<<"start_new_sequence done"<<"\n";

            fw_char_lstm.add_input(word_start);
            //cerr<<"added start of word symbol"<<"\n";
            /*for (unsigned j=0;j<w.length();j++){

                //cerr<<j<<":"<<w[j]<<"\n"; 
                Expression cj=lookup(*hg, char_emb, w[j]);
                fw_char_lstm.add_input(cj, hg);
        
               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }*/
	    std::vector<int> strevbuffer;
            for (unsigned j=0;j<ww.length();j+=UTF8Len(ww[j])){

                //cerr<<j<<":"<<w[j]<<"\n"; 
                std::string wj;
                for (unsigned h=j;h<j+UTF8Len(ww[j]);h++) wj+=ww[h];
                //std::cout<<"fw"<<wj<<"\n";
                int wjint=corpus.charsToInt[wj];
		//std::cout<<"fw:"<<wjint<<"\n";
		strevbuffer.push_back(wjint);
                Expression cj=lookup(*hg, char_emb, wjint);
                fw_char_lstm.add_input(cj);

               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }
            fw_char_lstm.add_input(word_end);
            //cerr<<"added end of word symbol"<<"\n";



            Expression fw_i=fw_char_lstm.back();

            //cerr<<"fw_char_lstm.back() done"<<"\n";

            bw_char_lstm.start_new_sequence();
            //cerr<<"bw start new sequence done"<<"\n";

            bw_char_lstm.add_input(word_end);
	    //for (unsigned j=w.length()-1;j>=0;j--){
            /*for (unsigned j=w.length();j-->0;){
               //cerr<<j<<":"<<w[j]<<"\n";
               Expression cj=lookup(*hg, char_emb, w[j]);
               bw_char_lstm.add_input(cj); 
            }*/

	    while(!strevbuffer.empty()) {
		int wjint=strevbuffer.back();
		//std::cout<<"bw:"<<wjint<<"\n";
		Expression cj=lookup(*hg, char_emb, wjint);
                bw_char_lstm.add_input(cj);
		strevbuffer.pop_back();
	    }
	    
            /*for (unsigned j=w.length()-1;j>0;j=j-UTF8Len(w[j])) {

                //cerr<<j<<":"<<w[j]<<"\n"; 
                std::string wj;
                for (unsigned h=j;h<j+UTF8Len(w[j]);h++) wj+=w[h];
                std::cout<<"bw"<<wj<<"\n";
                int wjint=corpus.charsToInt[wj];
                Expression cj=lookup(*hg, char_emb, wjint);
                bw_char_lstm.add_input(cj);

               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }*/
            bw_char_lstm.add_input(word_start);
            //cerr<<"start symbol in bw seq"<<"\n";     

            Expression bw_i=bw_char_lstm.back();

            vector<Expression> tt = {fw_i, bw_i};
            w=concatenate(tt); //and this goes into the buffer...
            //cerr<<"fw and bw done"<<"\n";
         }

	}
      /**************************************************/
      //cerr<<"concatenate?"<<"\n";

      /***************NO SPELLING*************************************/

      // Expression w = lookup(*hg, p_w, sent[i]);
      else { //NO SPELLING //(MT) this is where we do the lookup and take the word representation for the word of the sentence. This is the same here.
          //Don't use SPELLING
          //std::cout<<"don't use spelling"<<"\n";
          w=lookup(*hg, p_w, sent[i]);
      }

      Expression i_i;
      if (USE_POS) { //(MT) not sure if we want to have part-of-speech tags... probably not. Let's leave the code here just in case.
        Expression p = lookup(*hg, p_p, sentPos[i]);
        i_i = affine_transform({ib, w2l, w, p2l, p});
      } else {
        i_i = affine_transform({ib, w2l, w});
      }
      if (p_t && pretrained.count(raw_sent[i])) {
        Expression t = const_lookup(*hg, p_t, raw_sent[i]);
        i_i = affine_transform({i_i, t2l, t});
      }
      buffer[sent.size() - i] = rectify(i_i);
      bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree

    vector<Expression> output;
    vector<int> outputi;

    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM

    output.push_back(parameter(*hg, p_stack_guard));
    outputi.push_back(-999);  
    //same with output

    //not sure if i should take the dummy symbol out of the stack or not?

    stack_lstm.add_input(stack.back());
    output_lstm.add_input(output.back());

    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction, will also be used to restrict the number of actions allowed. if over 500, only shift and output allowed.
    unsigned eps_count = 0;
    unsigned copy_count = 0;
    while(stack.size() > 1 || buffer.size() > 1) { //(MT) main parsing loop. I think it is exactly the same in the MT model.
    //should we change the stack to > 1?  as if there is one element on the stack, it should still be outputed at some point

      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki, action_count, eps_count, copy_count))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back(), O, output_lstm.back()});
      Expression nlp_t = rectify(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      //can get rid of 569 to 583, keep action = correct_actions[action_count]
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned best_a = current_valid_actions[0];
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          best_a = current_valid_actions[i];
        }
      }

      unsigned action = best_a;
      string actionString=setOfActions[action];
      string word = "";
      vector<float> probs;
      if (actionString[0] == 'G' || actionString[0] == 'O') {
          int wordid = -1;
          float max_prob = -100000000;
          Expression probs_expr = cfsm->full_log_distribution(nlp_t);
          probs = as_vector(hg->incremental_forward());
          for (int i = 0; i < probs.size(); ++i) {
            if (probs[i] > max_prob) {
              max_prob = probs[i];
              wordid = i;
            }
          }
          word = corpus.out_d.Convert(wordid);
      }
      

      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        actionString=setOfActions[action];
        word = correct_words[action_count];
        if (best_a == action) { (*right)++; }
      }
      // do action
      actionString=setOfActions[action];
      //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];

      ++action_count;
      // action_log_prob = pick(adist, action)
      if (actionString[0] == 'G' || actionString[0] == 'O') {
        actionString = actionString + "(" + word + ")";
        log_probs.push_back(pick(adiste, action) - cfsm->neg_log_softmax(nlp_t, corpus.out_d.Convert(word)));
        Expression last_prob = 2*log_probs[log_probs.size() - 1];
	
        double probbb = as_scalar(hg->incremental_forward());
        assert(!boost::math::isinf(probbb));   

        Expression temp_sum = sum(log_probs);
	double lp = as_scalar(hg->incremental_forward());
        assert(!boost::math::isinf(lp));
        assert(!boost::math::isinf(-lp));
        results.push_back(actionString);
      }
      else {
        log_probs.push_back(pick(adiste, action));
        Expression last_prob = 2*log_probs[log_probs.size() - 1];
        double probbb = as_scalar(hg->incremental_forward());
        assert(!boost::math::isinf(probbb));

        results.push_back(actionString);
      }

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // get relation embedding from action (TODO: convert to relation from action?)
      Expression relation = lookup(*hg, p_r, action);


      char ac5 = ' ';
      if(actionString.size() > 3){
        ac5 = actionString[4];
      }


      if (ac =='S' && ac2=='H') {  // SHIFT //(MT) the same!
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
        eps_count = 0;
        copy_count = 0;

      } 
      else if (ac=='S' && ac2=='W'){ //SWAP --- Miguel //(MT) the same!
          assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

          //std::cout<<"SWAP: "<<"stack.size:"<<stack.size()<<"\n";

          Expression toki, tokj;
          unsigned ii = 0, jj = 0;
          tokj=stack.back();
          jj=stacki.back();
          stack.pop_back();
          stacki.pop_back();

          toki=stack.back();
          ii=stacki.back();
          stack.pop_back();
          stacki.pop_back();

          buffer.push_back(toki);
          bufferi.push_back(ii);

          stack_lstm.rewind_one_step();
          stack_lstm.rewind_one_step();


          buffer_lstm.add_input(buffer.back());

          stack.push_back(tokj);
          stacki.push_back(jj);

          stack_lstm.add_input(stack.back());

          //stack_lstm.rewind_one_step();
          //buffer_lstm.rewind_one_step();
          eps_count = 0;
          copy_count = 0;

	    }
      else if(ac == 'O' && ac5 != 'E'){  //OUTPUT (MT), this mirrors push, but is from stack to output
        assert(stack.size() > 1); // dummy symbol means > 1 (not >= 1)
        //this gets the symbol that is to be outputted
        unsigned out_int = corpus.out_d.Convert(actionString.substr(4, actionString.size() - 5));
        
        //this is the expression for the symbol that is being outputted
        Expression out_symbol = lookup(*hg, p_o, out_int);

        //then you push back the expression into the output, and add it to the output lstm
        output.push_back(out_symbol);
        output_lstm.add_input(out_symbol);
        stack.pop_back();
        stack_lstm.rewind_one_step();
        outputi.push_back(out_int);
        stacki.pop_back();
        ++eps_count;
        copy_count = 0;

      }
      else if(ac == 'D'){ //DELETE (mt), this deletes the top element of the buffer
        assert(buffer.size() > 1);
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        bufferi.pop_back();
        eps_count = 0;
        copy_count = 0;

      }
      else if(ac == 'C'){ //COPY, copies the top element of the buffer
        assert(buffer.size() > 1);
        buffer.push_back(buffer.back());
        buffer_lstm.add_input(buffer.back());
        bufferi.push_back(bufferi.back());
        eps_count = 0;
        ++copy_count;

      } //else add epsilon
      else{
        //this gets the symbol that is to be outputted
        unsigned out_int = corpus.out_d.Convert(actionString.substr(4, actionString.size() - 5));
        
        //this is the expression for the symbol that is being outputted
        Expression out_symbol = lookup(*hg, p_o, out_int);

        //then you push back the expression into the output, and add it to the output lstm
        output.push_back(out_symbol);
        output_lstm.add_input(out_symbol);
        outputi.push_back(out_int);
        eps_count = 0;
        copy_count = 0;
      }

  
  //this isn't needed in mt, leaving it as reference for now
  //delete after the other actions are confirmed working
  /*
      else { // LEFT or RIGHT //(MT) this needs to change! This is not only left or right, but output, etc.

        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        Expression dep, head;
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? dep : head) = stack.back();
        (ac == 'R' ? depi : headi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        (ac == 'R' ? head : dep) = stack.back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
        // composed = cbias + H * head + D * dep + R * relation
        Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
        Expression nlcomposed = tanh(composed);
        stack_lstm.rewind_one_step();
        stack_lstm.rewind_one_step();
        stack_lstm.add_input(nlcomposed);
        stack.push_back(nlcomposed);
        stacki.push_back(headi);
      }*/
    }
    assert(stack.size() == 1); // guard symbol, root
    assert(stacki.size() == 1);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    double lp = as_scalar(hg->incremental_forward());
    assert(!boost::math::isinf(lp));
    assert(!boost::math::isinf(-lp));

    return results;
  }

struct ParserState { //(MT) this is for beam-search... if we want to use it we will need another lstmbuilder for the output. For now, we do not want to use it.
  LSTMBuilder stack_lstm;
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;

  double score;
};

struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}

// run beam search //(MT) this is deprecated code. Forget about this for now.
vector<unsigned> log_prob_parser_beam(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with OOVs replaced
                     const vector<unsigned>& sentPos,
                     const vector<string>& setOfActions,
                     unsigned beam_size, double* log_prob) {
    abort();
#if 0
    vector<unsigned> results;
    ParserState init;

    stack_lstm.new_graph(hg);
    buffer_lstm.new_graph(hg);
    action_lstm.new_graph(hg);
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      i_p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      i_t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(i_action_start, hg);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence

    // precompute buffer representation from left to right
    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      Expression w = lookup(*hg, p_w, sent[i]);
      Expression i;
      if (USE_POS) {
        Expression p = lookup(*hg, p_p, sentPos[i]);
        i_i = hg->add_function<AffineTransform>({i_ib, i_w2l, i_w, i_p2l, i_p});
      } else {
        i_i = hg->add_function<AffineTransform>({i_ib, i_w2l, i_w});
      }
      if (p_t && pretrained.count(raw_sent[i])) {
        Expression t = hg->add_const_lookup(p_t, sent[i]);
        i_i = hg->add_function<AffineTransform>({i_i, i_t2l, i_t});
      }
      Expression inl = hg->add_function<Rectify>({i_i});
      buffer[sent.size() - i] = i_inl;
      bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b, hg);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back(), hg);

    init.stack_lstm = stack_lstm;
    init.buffer_lstm = buffer_lstm;
    init.action_lstm = action_lstm;
    init.buffer = buffer;
    init.bufferi = bufferi;
    init.stack = stack;
    init.stacki = stacki;
    init.results = results;
    init.score = 0;
    if (init.stacki.size() ==1 && init.bufferi.size() == 1) { assert(!"bad0"); }

    vector<ParserState> pq;
    pq.push_back(init);
    vector<ParserState> completed;
    while (pq.size() > 0) {
      const ParserState cur = pq.back();
      pq.pop_back();
      if (cur.stack.size() == 2 && cur.buffer.size() == 1) {
        completed.push_back(cur);
        if (completed.size() == beam_size) break;
        continue;
      }

      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], cur.buffer.size(), cur.stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression p_t = hg->add_function<AffineTransform>({i_pbias, i_S, cur.stack_lstm.back(), i_B, cur.buffer_lstm.back(), i_A, cur.action_lstm.back()});

      // nlp_t = tanh(p_t)
      Expression nlp_t = hg->add_function<Rectify>({i_p_t});

      // r_t = abias + p2a * nlp
      Expression r_t = hg->add_function<AffineTransform>({i_abias, i_p2a, i_nlp_t});

      //cerr << "CVAs: " << current_valid_actions.size() << " (cur.buf=" << cur.bufferi.size() << " buf.sta=" << cur.stacki.size() << ")\n";
      // adist = log_softmax(r_t)
      hg->add_function<RestrictedLogSoftmax>({i_r_t}, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());

      for (auto action : current_valid_actions) {
        pq.resize(pq.size() + 1);
        ParserState& ns = pq.back();
        ns = cur;  // copy current state to new state
        ns.score += adist[action];
        ns.results.push_back(action);

        // add current action to action LSTM
        Expression action = lookup(*hg, p_a, action);
        ns.action_lstm.add_input(i_action, hg);

        // do action
        const string& actionString=setOfActions[action];
        //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
        const char ac = actionString[0];
        if (ac =='S') {  // SHIFT
          assert(ns.buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          ns.stack.push_back(ns.buffer.back());
          ns.stack_lstm.add_input(ns.buffer.back(), hg);
          ns.buffer.pop_back();
          ns.buffer_lstm.rewind_one_step();
          ns.stacki.push_back(cur.bufferi.back());
          ns.bufferi.pop_back();
        } else { // LEFT or RIGHT
          assert(ns.stack.size() > 2); // dummy symbol means > 2 (not >= 2)
          assert(ac == 'L' || ac == 'R');
          Expression dep, head;
          unsigned depi = 0, headi = 0;
          (ac == 'R' ? dep : head) = ns.stack.back();
          (ac == 'R' ? depi : headi) = ns.stacki.back();
          ns.stack.pop_back();
          ns.stacki.pop_back();
          (ac == 'R' ? head : dep) = ns.stack.back();
          (ac == 'R' ? headi : depi) = ns.stacki.back();
          ns.stack.pop_back();
          ns.stacki.pop_back();
          // get relation embedding from action (TODO: convert to relation from action?)
          Expression relation = lookup(*hg, p_r, action);

          // composed = cbias + H * head + D * dep + R * relation
          Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
          // nlcomposed = tanh(composed)
          Expression nlcomposed = tanh(composed);
          ns.stack_lstm.rewind_one_step();
          ns.stack_lstm.rewind_one_step();
          ns.stack_lstm.add_input(i_nlcomposed, hg);
          ns.stack.push_back(i_nlcomposed);
          ns.stacki.push_back(headi);
        }
      } // all curent actions
      prune(pq, beam_size);
    } // beam search
    assert(completed.size() > 0);
    prune(completed, 1);
    results = completed.back().results;
    assert(completed.back().stack.size() == 2); // guard symbol, root
    assert(completed.back().stacki.size() == 2);
    assert(completed.back().buffer.size() == 1); // guard symbol
    assert(completed.back().bufferi.size() == 1);
    *log_prob = completed.back().score;
    return results;
#endif
  }
};

//(MT) still useful.
void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

unsigned compute_correct(const map<int,int>& ref, const map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}

//(MT) we do not output any conll here. So we need to come up with the correct reordering output.
void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings, 
                  const map<unsigned, string>& intToWords, 
                  const vector<string>& setOfActions,
                  const vector<string>&  actions) {

      int sent_len = sentence.size();
      vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
      vector<string> outputi;

      for (unsigned i = 0; i < sent_len; ++i)
        bufferi[sent_len - i] = sentence[i];
      bufferi[0] = -999;
      for (string action: actions) { // loop over transitions for sentence
        const string& actionString=action;
        const char ac = actionString[0];
        const char ac2 = actionString[1];
        char ac5 = ' ';
        if(actionString.size() > 3){
          ac5 = actionString[4];
        }
        if (ac =='S' && ac2=='H') {  // SHIFT //(MT) the same!
          assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
            stacki.push_back(bufferi.back());
            bufferi.pop_back();
        } 
        else if (ac=='S' && ac2=='W'){ //SWAP --- Miguel //(MT) the same!
          assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
          unsigned ii = 0, jj = 0;
          jj=stacki.back();
          stacki.pop_back();
          ii=stacki.back();
          stacki.pop_back();
          bufferi.push_back(ii);

          stacki.push_back(jj);
        }
        else if(ac == 'O' && ac5 != 'E'){  //OUTPUT (MT), this mirrors push, but is from stack to output
          assert(stacki.size() > 1); // dummy symbol means > 1 (not >= 1)
          string word = actionString.substr(4, actionString.size() - 5);
          outputi.push_back(word);
          stacki.pop_back();
        }
        else if(ac == 'D'){ //DELETE (mt), this deletes the top element of the buffer
          assert(bufferi.size() > 1);
          bufferi.pop_back();
        }
        else if(ac == 'C'){ //COPY, copies the top element of the buffer
          assert(bufferi.size() > 1);
          bufferi.push_back(bufferi.back());
        } //else add epsilon
        else {
          string word = actionString.substr(4, actionString.size() - 5);
          outputi.push_back(word);
        }
      }

      for (string out : outputi) {
        cout << out << " ";
      }
  cout << endl;

}

//(MT) main method. Looks like we can reuse most of the code.
int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");

  USE_SPELLING=conf.count("use_spelling"); //Miguel
  corpus.USE_SPELLING=USE_SPELLING;

  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  REL_DIM = conf["rel_dim"].as<unsigned>();
  const unsigned beam_size = conf["beam_size"].as<unsigned>();
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << "parser_" << (USE_POS ? "pos" : "nopos")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << "-pid" << getpid() << ".params";


  int best_correct_heads = 0;
  double best_perplexity = 100000000000;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;
  bool softlinkCreated = false;
  corpus.load_correct_actions(conf["training_data"].as<string>());	
  const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
  kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);
  kEPSILON = corpus.get_or_add_word(EPSILON);
  cerr << "nwords: " << corpus.nwords << endl;
  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    pretrained[kEPSILON] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
      if (corpus.wordsToInt[word] != 0) {
         unsigned id = corpus.get_or_add_word(word);
         pretrained[id] = v;
      }

    }
  }

  cerr << "nwords: " << corpus.nwords << endl;

  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;
  {  // compute the singletons in the parser's training data
    map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto word : sent.second) { training_vocab.insert(word); counts[word]++; }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  cerr << "Number of words: " << corpus.nwords << endl;
  VOCAB_SIZE = corpus.nwords + 1;
  //VOCAB_SIZE = 258766;
  cerr << "Number of UTF8 chars: " << corpus.maxChars << endl;
  if (corpus.maxChars>255) CHAR_SIZE=corpus.maxChars;
  OUTPUT_DIM = corpus.out_d.size();
  ACTION_SIZE = corpus.nactions + 1;
  //POS_SIZE = corpus.npos + 1;
  POS_SIZE = corpus.npos + 10;
  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;

  Model model;
  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, conf["clusters"].as<string>(), &corpus.out_d, &model);


  // OOV words will be replaced by UNK tokens
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  if (USE_SPELLING) VOCAB_SIZE = corpus.nwords + 1;
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    cerr << "Training started."<<"\n";
    vector<unsigned> order(corpus.nsentences);
    for (unsigned i = 0; i < corpus.nsentences; ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
    unsigned si = corpus.nsentences;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    while(!requested_stop) {
      ++iter;
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.nsentences) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             //cerr << "**SHUFFLE\n";
             //random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           const vector<unsigned>& sentence=corpus.sentences[order[si]];
           vector<unsigned> tsentence=sentence;
           if (unk_strategy == 1) {
             for (auto& w : tsentence)
               if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
           }
	   const vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]]; 
	   const vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,actions,corpus.actions,corpus.intToWords, corpus.correct_act_words[order[si]], &right);
           double lp = as_scalar(hg.incremental_forward());
           //total_perplexity += exp(-lp/possible_actions.size());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
      }
      //sgd.status();
      //cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentences) << ")\tperplexity: "<< total_perplexity/status_every_i_iterations << endl;
      llh = trs = right = 0;

      static int logc = 0;
      ++logc;

      //this needs to change based on how we are doing the evaluation
      //not really sure how to change it, though...
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = corpus.nsentencesDev;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        double correct_heads = 0;
        double new_perplexity = 0;
        double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const vector<unsigned>& sentence=corpus.sentencesDev[sii];
	   const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
	   const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
           vector<unsigned> tsentence=sentence;
	   if (!USE_SPELLING) {
                for (auto& w : tsentence)
                    if (training_vocab.count(w) == 0) w = kUNK;
           }

           ComputationGraph hg;
	         vector<string> pred = parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords, vector<string>(), &right);
           double lp = as_scalar(hg.incremental_forward());
           new_perplexity += exp(lp/possible_actions.size());
	   //double lp = 0;
           //vector<unsigned> pred = parser.log_prob_parser_beam(&hg,sentence,sentencePos,corpus.actions,beam_size,&lp);

           //shouldn't need this, commented out for now-Dan
           /*
           llh -= lp;
           trs += actions.size();
           map<int,int> ref = parser.compute_output(sentence.size(), actions, corpus.actions);
           map<int,int> hyp = parser.compute_output(sentence.size(), pred, corpus.actions);
           //output_conll(sentence, corpus.intToWords, ref, hyp);
           correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           total_heads += sentence.size() - 1;
           */
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        //remade output for perplexity
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ") perplexity=" << new_perplexity/dev_size << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
      
        //this uses perplexity instead of best_heads to save model
        if(new_perplexity < best_perplexity){
          best_perplexity = new_perplexity;

          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }

        //this isn't needed, keeping it around in case my changes break it and i need to reference it
        /*
        if (correct_heads > best_correct_heads) {
          best_correct_heads = correct_heads;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        } */
      }
    }
  } // should do testing?
  if (true) { // do test evaluation
    double llh = 0;
    double trs = 0;
    double right = 0;
    double correct_heads = 0;
    double total_heads = 0;
    double total_perplexity = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.nsentencesDev;
    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const vector<unsigned>& sentence=corpus.sentencesDev[sii];
      const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
      const vector<string>& sentenceUnkStr=corpus.sentencesStrDev[sii]; 
      const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      vector<unsigned> tsentence=sentence;
      if (!USE_SPELLING) {
        for (auto& w : tsentence)
	  if (training_vocab.count(w) == 0) w = kUNK;
      }
      ComputationGraph cg;
      double lp = 0;
      vector<string> pred;
      if (beam_size == 1)
        pred = parser.log_prob_parser(&cg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords, vector<string>(), &right);
      //else
      //  pred = parser.log_prob_parser_beam(&cg,sentence,tsentence,sentencePos,corpus.actions,beam_size,&lp);
      
      double ls = as_scalar(cg.incremental_forward());

      total_perplexity += exp(ls/possible_actions.size());

      output_conll(tsentence, sentencePos, sentenceUnkStr, corpus.intToWords, corpus.actions, pred);
    }
    double perplexity = total_perplexity/corpus_size;

    
    auto t_end = std::chrono::high_resolution_clock::now();
    cerr << "TEST average perplexity = " << perplexity << "\t[" << corpus_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
  }
  for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    //cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose() << endl;
    //cerr << corpus.actions[i] << '\t' << parser.p_p2a->values.col(i).transpose() << endl;
  }
}
