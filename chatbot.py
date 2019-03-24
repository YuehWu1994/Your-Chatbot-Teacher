#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#just for James
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Dense, TimeDistributed
from keras.models import Model
from keras.layers import Embedding
from keras.layers import GRU, Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import model_from_json
# from lstmEnc_DNN import lstmEncoder 
from encoder_decoder_handler import encoder_decoder_handler
from evaluate import countBLEU
import keras.utils as ku
import copy
import math


class Chatbot:


    #hyperparameters
    batch_size = 0

    #data loading methods
    encoder_decoder_handler = None

    #classifier model object
    classifier_model = None
    max_train_length = 20

    #data
    train_X = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    embedding_matrix = []


    def __init__(self):
        self.batch_size = 50

        self.encoder_decoder_handler = encoder_decoder_handler(self.batch_size)


    # builds seq2seq model with encoder and decoder
    def build_model(self):
        latent_dim = 256


        #initializes encoder
        word_vec_input = Input(shape=(None,), name="input_1")
        encoder_embed = Embedding(input_dim=self.encoder_decoder_handler.vocab_size, output_dim=100, weights=[self.embedding_matrix], input_length=self.max_train_length)(word_vec_input)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_embed)
        encoder_states = [state_h, state_c]

        #initializes decoder
        decoder_inputs = Input(shape=(None, self.encoder_decoder_handler.vocab_size), name="input_3")
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.encoder_decoder_handler.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        encoder_decoder = Model([word_vec_input, decoder_inputs], decoder_outputs)


        #compiles model
        encoder_decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



        # define encoder inference model
        inference_encoder_model = Model(word_vec_input, encoder_states)


        # define decoder inference model
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        inference_decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)


        print(encoder_decoder.summary())

        return encoder_decoder, inference_encoder_model, inference_decoder_model


    #returns an inference model that can be used for predicting one word at a time
    def build_inference_model(self, encoder_decoder_model):
        #Defining the inference models requires reference to elements of the model used for training in the example. 
        #Alternately, one could define a new model with the same shapes and load the weights from file.

        pass



    def predict_sequence(self, inference_model, X1, X2, n_steps, cardinality):
        #encode
        X2 = np.reshape(X2, (1, len(X2)))
        state = [X2]   

        output = list()
        for t in range(n_steps):
            
            # predict next word
            x = np.reshape(embedding_matrix[X1[t]], (1,1,100))
            yhat, h = inference_model.predict([x] + state)
            # store prediction
            output.append(yhat[0,0,:])
            # update state
            state = [h]
            # update target sequence by next word
            #target_seq = X1[t+1]
        return np.array(output)

    def one_hot_decode(self, encoded_seq):
    	return [np.argmax(vector) for vector in encoded_seq]



    def interpret(self, lstm, y, target):
        ans = ""
        predSeq = ""
        for i in range(lstm.max_train_len):
            ans +=  "<UNK> " if (y[i] == 0) else (lstm.index_word[y[i]] + ' ')
            predSeq +=  "<UNK> " if (target[i] == 0) else (lstm.index_word[target[i]] + ' ')
        print(ans)
        print(predSeq)



    #loads data from the lstm_encoder class
    def load_data(self, num_data_points):

        X_train, y_train, X_test, y_test, self.embedding_matrix = self.encoder_decoder_handler.create_Emb(num_data_points)

        #3D vectorized representation of words
        new_X_train = np.zeros((len(X_train), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_y_train = np.zeros((len(y_train), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_X_test = np.zeros((len(X_test), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')
        new_y_test = np.zeros((len(y_test), self.max_train_length, self.encoder_decoder_handler.vocab_size), dtype='int8')


        ### converts [5, 12, 1, 482, 8] into a list of binary lengths with 1s at specific indices ###
        #converts training data
        for x in range(0, len(X_train)):
            for y in range(0, len(X_train[x])):
                x_train_index = X_train[x][y]
                y_train_index = y_train[x][y]

                new_X_train[x][y][x_train_index] = 1
                new_y_train[x][y][y_train_index] = 1

        #converts testing data
        for x in range(0, len(X_test)):
            for y in range(0, len(X_test[x])):
                x_test_index = X_test[x][y]
                y_test_index = y_test[x][y]

                new_X_test[x][y][x_test_index] = 1
                new_y_test[x][y][y_test_index] = 1
        self.decoder_X_train = new_X_train
        self.decoder_y_train = new_y_train
        self.decoder_X_test = new_X_test
        self.decoder_y_test = new_y_test

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



        print("lengths of data: ")
        print("X_train: "+str(len(self.X_train)))
        print("y_train: "+str(len(self.y_train)))
        # print("X_val: "+str(len(self.X_val)))
        # print("y_val: "+str(len(self.y_val)))
        print("X_test: "+str(len(self.X_test)))
        print("y_test: "+str(len(self.y_test)))
        print("embedding matrix: "+str(len(self.embedding_matrix)))
        print()

    def predict_sequence(self, inference_encoder_model, inference_decoder_model, input_sequence):

        #converts hot one vector into a string sentence
        original_sequence = self.encoder_decoder_handler.decode_sequence(input_sequence[0])

        print("Input sequence: "+str(original_sequence))


        # Encode the input as state vectors.
        states_value = inference_encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.encoder_decoder_handler.get_token_index("anime")] = 1.



        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        predicted_sequence = []
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = inference_decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.encoder_decoder_handler.decode_int(sampled_token_index)

            decoded_sentence += sampled_char+" "

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == 'eos' or len(decoded_sentence) > 50):
                stop_condition = True

            #Update the target sequence (of length 1).
            target_seq = np.array([np.concatenate((target_seq[0], np.zeros((1, self.encoder_decoder_handler.vocab_size))))])
            target_seq[0, -1, sampled_token_index] = 1.

            #resets the target sequence each time
            # target_seq = np.zeros((1, 1, self.encoder_decoder_handler.vocab_size))
            # target_seq[0, 0, sampled_token_index] = 1.

            predicted_sequence.append(sampled_token_index)

            # Update states
            states_value = [h, c]



        print("Decoded sequence: "+str(decoded_sentence))
        return predicted_sequence


    def evaluate_sequences(self):

        sentences = ["sos agil in sao i personally find him the only one in aincrad arc who has a brain eos", 
                    "sos unravel kukuku eos", 
                    "sos damn this looks good eos",
                    "and tranquil also interesting how the title is in hiragana instead of katakana considering it s an english title eos",
                    "been about a year and a half since i watched the first season so this is good for me eos",
                    "would ve been a few hours flight from japan to egypt into a journey that takes them 50 days eos",
                    "nt think i watched more currently airing stuff until durarara and the tatami galaxy aired like a year later eos",
                    "any one else think this or can confirm or deny any common key factors thanks again pittman for hosting eos",
                    "sos left side best except for rider got ta vote for rider eos",
                    "no tabi last exile and steamboy also the legend of korra if it does nt have to be anime eos",
                    "sos amazing article a lot of the points are the reasons why i love one piece so much eos",
                    "sos kurosaki ichigo http images5fanpopcomimagephotos29000000ichigowallpaperkurosakiichigo290694271024768jpg and kurosaki mea http staticzerochannetkurosakimeafull1689483jpg eos",
                    "sos as much as i do nt like recommending it rebuild of evangelion specifically 20 defrosts rei a lot eos",
                    "sos accelerator reason one way road eos",
                    "sos pina co lada kanie aisu kyuubu latifah eos",
                    "when you see an anime about cute little girls fighting in tanks do you have any other good example eos",
                    "sos pina co lada kanie aisu kyuubu latifah eos",
                    "time ever a couple of hours ago here s the thread https wwwredditcomrjapancomments464491the mythology behind ringu looking for movies eos",
                    "sos crappily designed amagamithemed one that i should probably update but im way too lazy http myanimelistnetanimelistpeengwin eos",
                    "why well it is nt like we are going to get the rest of the canon any time soon eos",
                    "sos kanie soya and sento isuzu hands down eos",
                    "https youtubei6fwe3eftrq will forever remain on my shit list some people say it grew on them but for me eos",
                    "rewatch good on you everyone who watched it all and especially good on you uspiranix for setting this up eos",
                    "really confused cause i ve never heard anything like it but what the fuck sort of accent is that eos",
                    "and causes people to just care for each other and do what s right at least for a time eos",
                    "sos obligatory dango family song https wwwyoutubecomwatch v 0i6yfdgdoc4 hurrystarfish song https wwwyoutubecomwatch v 3 xhisyl9aq clannad spoilers eos",
                    "bakemonogatari but my english was nt good enough in the end umineko was shit but the oped were excellent eos",
                    "sos time travel udastales will hate it eos",
                    "sos probably accelerator s ability from index eos",
                    "lighter note bibi from love live is my shit right now idol hell is a horrific yet wonderful place eos",
                    "sos this ed from kaminomi right here https wwwyoutubecomwatch v vxinraldqhk edit spelling eos",
                    "too out of the question while conveying so much really though the achievement was that ending they did it eos",
                    "way through nana the drama can be a little too much for me sometimes but i enjoy it overall eos",
                    "sos a news app http myanimelistnetanime30721hacka doll the animation eos",
                    "sos wait you re not jordy sublimer flyingbunsofdoom eos",
                    "no victims today yey in memoriam wolfgang grimmer and yamato ishida out on the nomination phase i know unbelivable eos",
                    "honestly this could apply to every it s about some girls in high school show that s actually funny eos",
                    "exercise has 37 senses many secret maid guy abilities including usb data transfer proudly wears a maid uniform kukukukuku eos",
                    "sos woooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo eos",
                    "here http myanimelistnetanimelistthe shaqtus i m looking for something to watch next i m open to pretty much anything eos",
                    "sos sekkou boys can someone shop saitma into the group xd eos",
                    "lighting there is more to her hair than what you drew i think not sure though otherwise great job eos",
                    "sos myanimelist http myanimelistnetanimelistrxhysteria status 7amp order 4 just a simple design with fma background eos",
                    "no victims today yey in memoriam wolfgang grimmer and yamato ishida out on the nomination phase i know unbelivable eos",
                    "they are the worst part of the story and possibly one of the only bad writing in the show eos",
                    "do nt think she typifies the trope at least edit also of course shiraishi is best girl please op eos",
                    "http myanimelistnetanime22055gakkou no kaidan recaps anyone have any idea of its existence or is this a made up title eos",
                    "his bike so he does nt try killing himself animated for such a serious moment it made me smile eos",
                    "sos i think you forgot about best girl hana http cdnmyanimelistnetscommonuploaded files1443544738ceca3f6530f906e3f13e6ba3458b37ccjpeg eos",
                    "sos gt ctrlf space dandy gt no results wot eos"]

# Test data
# Should have responded with: an ally a young man named mitsuo oh and mc s name is nobu uncle s name is mikio eos
# Should have responded with: all last summer boy would i write walls of text after every episode now sounds fun let s go eos
# Should have responded with: sos the awful phantom requiem for the phantom 2nd op https wwwyoutubecomwatch v xf2ynhaeqvk eos
# Should have responded with: sos vote for the king of conquerors he s riding straight to first place eos
# Should have responded with: case anyone is interested spice and wolf is getting a sequel for its 10th anniversary this year http wwwanimenewsnetworkcomnews20160209spiceandwolflightnovelseriesgetssequelon10thanniversary98489 eos
# Should have responded with: and seeing hikaru stirs up her feelings again and hikaru s is nt that just what everybody needs now eos
# Should have responded with: both her ark in kajitsu and in general in rakuen even more so if you play the visual novel eos
# Should have responded with: sos dancing to platinum disco on full blast during stops eos
# Should have responded with: sos this is a public service announcement fuck kamina vote dio that is all eos
# Should have responded with: the tamako rewatch or a bad pun just fill this thread with all the cool stuff you really like eos
# Should have responded with: sos i was nt the biggest fan of rokka no yuusha but i really liked fremy and hans eos
# Should have responded with: he is into birdwomen the second has his fiancee died and became a ghost the last is a lesbian eos
# Should have responded with: sos less than three months since the last episode this ca nt be right that s too soon eos
# Should have responded with: avoid getting double references i ve also watched magi bleach trigun and death note realllly loved the last two eos
# Should have responded with: sos miami guns http myanimelistnetanime783miami guns maybe eos
# Should have responded with: sos at first i thought it was some kind of watchmojo anime video eos
# Should have responded with: sos right in the middle of finals but 100 worth it looking forward to this eos
# Should have responded with: wwwyoutubecomwatch v 8dkljqhcsoe just not any day i m listening to it worst gintama op by a long stretch eos
# Should have responded with: a very likable person early on this or the following episode should be a turning point in the series eos
# Should have responded with: what shows should i watch next how much should i be watching daily in order to keep up thanks eos
# Should have responded with: her paranoia agent s the show became so successful that you can actually buy maromi merchandise in real life eos
# Should have responded with: time i ll make sure to nominate gintama s gintoki and sket dance s gintoki for all i care eos
# Should have responded with: political and conspiracy driven shows make me a house of cards esque anime show and i would be happy eos
# Should have responded with: sos cram school https youtubeafbns gkhpw so good but makes me sad eos
# Should have responded with: assumed it was going to be a generic high school harem romcom but it turned out to be amazing eos
# Should have responded with: incapable of pronouncing ein and zwei correctly do nt remember for which one i ve settled in the end eos
# Should have responded with: sos nodame cantabile cowboy bebop and of course trigun first animes i watched still my favorites eos
# Should have responded with: sos vote gon that is all eos
# Should have responded with: sos now this is my kind of lewd hunchedover eos
# Should have responded with: time for what i actually wanted to see i liked her fine when she was still a side character eos
# Should have responded with: to read more i m already excited for the next part and i ve never read anything from them eos
# Should have responded with: sos boys prepare ya ll tomorrow begins the battle to take shinji to victory excitedyui eos
# Should have responded with: lovers and people who are yet to watch it not even hitler would be able to hate that anime eos
# Should have responded with: enjoy american animated shows almost as equally as i enjoy anime but the majority record similarly as japan does eos
# Should have responded with: sos mgronald s eos
# Should have responded with: even if quite foolish sometimes is a really soft and kindhearted guy vote for best husbando http iimgurcome47sngdjpg 1 eos
# Should have responded with: m not doing it again it s not worth it and i m pretty much sub only these days eos
# Should have responded with: shinsekai yori http myanimelistnetanime13125shinsekai yori q shinsekai 20yori trust me that the school setting is very different than usual eos
# Should have responded with: of dubs are nt good but madoka thankfully got a good one an i actually prefer the dub here eos
# Should have responded with: ears in the later episodes no comment about the op and ed since i already forget what they are eos
# Should have responded with: sos oregairu because then society can hate society too eos
# Should have responded with: sos how much extra would a custom made one cost eos
# Should have responded with: office supplies runnerup i passed a toshino kyoko https wwwyoutubecomwatch v nzgqedw7vj0 cosplayer and yelled her name at her eos
# Should have responded with: sos gt wild dance of kyoto anime initially read that as kyoani got my hopes up lol nope giveuponlife eos
# Should have responded with: why so many people think its economics rather than just barter the show its on hold with 3 episodes eos
# Should have responded with: with dereban click at your own risk http jenlogtumblrcompost98576863914kamiaidouinabahimekodereban speaking of which i really need to rewatch that show eos
# Should have responded with: sos goddammit every rewatch happens right after i finish a show eos
# Should have responded with: be immersed in the story sadly it s only two episodes and the novel s translation are barely breathing eos
# Should have responded with: an unwelcome group of admirers and wishes he could return to normal life as a high schooler the end eos
# Should have responded with: w 776 terrible womanizer https mediagiphycommedia36tjhkehimdsagiphygif fellow donut lover http 49mediatumblrcoma05cd4405328a2a353a99e44e41e0d1btumblr ms5cofssjh1szqepgo1 500gif quite the intimidator http iimgurcomez4gznbwebm plz eos



# Test data
# Input sequence: a child he then tries to make everything right but just causes more problem it would be a dramedy eos
# Decoded sequence: an not episode and

# Input sequence: have any info about if they re going to sell the whole collection of light novels as a package eos
# Decoded sequence: sos was show ever

# Input sequence: kotourasan and friends go to sing karaoke https youtubecdjcuhuosci t 14s and she tries to sing her own op eos
# Decoded sequence: sos awful my where

# Input sequence: voted for james over guy at least it makes it even easier for rider to ride to the top eos
# Decoded sequence: sos i actually was eos

# Input sequence: assumed it was going to be a generic high school harem romcom but it turned out to be amazing eos
# Decoded sequence: sos ange but the

# Input sequence: wish someone would shoot hikaru so i would nt have to sit here and watch him be an idiot eos
# Decoded sequence: and very my is that

# Input sequence: oreshura best girl http mdzanimemewpcontentuploads201302124jpg kuudere shares the same voice actor as hitagi edit she sounds like senjougahara xd eos
# Decoded sequence: sos ark of in   eos

# Input sequence: no one tripped and grabbed my boobs by accident though so it could be a lot worse more weeb eos
# Decoded sequence: sos was a finale and

# Input sequence: everyone they are up against are like my least favourite characters from their respective shows ohh well vote oreki eos
# Decoded sequence: sos is a other

# Input sequence: sos allucard case closed eos
# Decoded sequence: sos was my to favorite

# Input sequence: to admit that sora and jibril are fun characters to watch especially jibril and i love her english va eos
# Decoded sequence: sos was the but this

# Input sequence: follow our mathematics discoveries timeline hell i do nt care if this newton suddenly talks about boolean algebra lol eos
# Decoded sequence: sos story the was

# Input sequence: shot up at the end of the card game was your character popping a soul item in dark souls eos
# Decoded sequence: sos than since   eos

# Input sequence: so if you ve watched anything similar let me know thanks here s my mal http myanimelistnetanimelistdingadong status 7 eos
# Decoded sequence: avoid on is the

# Input sequence: sos something like old miami vice that just convey the feeling of 80s miami eos
# Decoded sequence: sos guns of i

# Input sequence: sos i thought the title said noises boy did i get an interesting surprise eos
# Decoded sequence: sos started and you

# Input sequence: sos sequel hype eos
# Decoded sequence: sos a the 100 forward

# Input sequence: remember a specific track i hated the most but here s just some random one https wwwyoutubecomwatch v fphb63utz5w eos
# Decoded sequence: wwwyoutubecomwatch was

# Input sequence: ll fly right by i am really enjoying the dogfight like maneuvers constantly being put on by this show eos
# Decoded sequence: a likable bad great

# Input sequence: place in 80 s s cali rather than florida here is a youtube review https wwwyoutubecomwatch v rbrvp3 zjw8 eos
# Decoded sequence: sos this thread in a

# Input sequence: tell me which chapter i should begin reading from also does onizuka officially start dating fuyutsuki in the manga eos
# Decoded sequence: sos are of i

# Input sequence: m hoping that levi just being from attack on titan will be enough to overthrow kamina because fuck kamina eos
# Decoded sequence: sos ll up the if

# Input sequence: ll post my own idea as a reply to the topic to keep this post as short as possible eos
# Decoded sequence: political conspiracy as the show eos

# Input sequence: sos a card game eos
# Decoded sequence: sos a my me

# Input sequence: sos haikyuu its pretty cliche for the sports genre but the characters make it work eos
# Decoded sequence: assumed was about harem even eos

# Input sequence: through and i liked it a lot better easily the worst dub i ve heard by funimation good show eos
# Decoded sequence: sos pronouncing anime me s

# Input sequence: the entire cast of legend of the galactic heroes http digitalconfederacycomimagesarticlesurrdayreviewsloghloghcharacterspng and pretty much everything else about the show eos
# Decoded sequence: sos cantabile a i

# Input sequence: that voted for gilgamesh explain why you chose him over accelerator i really want to know salt level 420 eos
# Decoded sequence: sos started you

# Input sequence: performed automatically please contact the moderators of this subreddit messagecompose to ranime if you have any questions or concerns eos
# Decoded sequence: sos on the

# Input sequence: sos i love psycho pass but i m not too fond of akane eos
# Decoded sequence: time here wanted i time

# Input sequence: sos uchuu senkan yamato looks interesting eos
# Decoded sequence: to about this next never eos

# Input sequence: sos gt souma vs vash why do you keep doing this to me eos
# Decoded sequence: sos prepare tomorrow for this

# Input sequence: commitment to the close up on renge s face for such a prolonged scene was amazing i need crayons eos
# Decoded sequence: lovers people list not be

# Input sequence: sos i would recommend sore ga seiyuu http myanimelistnetanime29163sore ga seiyuu for some insight eos
# Decoded sequence: enjoy animated     the

# Input sequence: sos for me it s my neighbor pedoro https youtubewnkipydwzy eos
# Decoded sequence: sos s

# Input sequence: rip best level five http iimgurcomnr4r6pxjpg at least you lost to an opponent with fucking fabulous hair http iimgurcomgmzi3kngifv eos
# Decoded sequence: sos quite my her

# Input sequence: of dubs are nt good but madoka thankfully got a good one an i actually prefer the dub here eos
# Decoded sequence: m doing m

# Input sequence: sos ergo proxy jinrou psychopass bersek baccano eos
# Decoded sequence: shinsekai yori the world http

# Input sequence: showed up and i said fuck that to the dub voice and rode the sub out to the end eos
# Decoded sequence: of are madoka but   eos

# Input sequence: serious arc edit 2 uhh i do nt really know why i m getting down voted that hard x eos
# Decoded sequence: ears watching of know to

# Input sequence: act like military professional compassionate human beings trying to do the right thing when possible i love this series eos
# Decoded sequence: sos watched a eos

# Input sequence: was definitely not expecting a sculpture fanart looks like it was worth all your timeeffort as this looks stunning eos
# Decoded sequence: sos much would eos

# Input sequence: i read in a very vivid dream i had lately that s as hardcore as it gets i guess eos
# Decoded sequence: office i pretty such   eos

# Input sequence: sos oreun ji no ni hayaku eos
# Decoded sequence: sos was my to time

# Input sequence: too busy to watch again but i can repost my ln comparisons from last year if anyone s interested eos
# Decoded sequence: clock show its of

# Input sequence: sos so many shows in the thread only one show i have nt seen eos
# Decoded sequence: with click this need to

# Input sequence: sos right in the middle of finals but 100 worth it looking forward to this eos
# Decoded sequence: sos are right

# Input sequence: sos this thread https wwwredditcomranimecomments431emttell me about your favorite anime that is ranked might interest you eos
# Decoded sequence: be it all of

# Input sequence: sos nonhentai anime either about cute milfscakesols doing cute things or romcomdrama harem with milfscakesols eos
# Decoded sequence: an group wishes but the

# Input sequence: sos lancer ga shinda eos
# Decoded sequence: w minutes real of http



        replies = ["sos i did nt like oreimo but kuroneko was good eos",
                    "ears in the later episodes no comment about the op and ed since i already forget what they are eos",
                    "sos hype is real i was hoping this would get an adaption eos",
                    "sos great job i would definitely buy one if i had the money eos",
                    "sos excited for this can never have enough holo in my life eos",
                    "to see how french frenchington ends up joining the team and if they ever get to finish their meal eos",
                    "blood and yowamushi pedal at varying points and did nt finish them until a while after they had finished eos",
                    "and they are willing to understand once sora proves herself the characters are all real people not including fool eos",
                    "so maybe i can get some ideas from all of those who have already made their new style c eos",
                    "sos fma and gosick eos",
                    "again and i ll feel like i m back in the thick of the narrative such a wonderful manga eos",
                    "ton of koutarous but the presence of one https smediacacheak0pinimgcomoriginals1219ed1219ed717fc2bfce372759bba2fe1cfegif is enough to make it the most interesting party eos",
                    "do nt think she typifies the trope at least edit also of course shiraishi is best girl please op eos",
                    "sos shinsuke takasugi why s an eye for an eye eos",
                    "sos straight witches http iimgurcomwpovkuwjpg in outbreak company as a reference to strike witches eos",
                    "they wrote six potential plotprogression ideas and then rolled a die to determine which one they d go with eos",
                    "sos straight witches http iimgurcomwpovkuwjpg in outbreak company as a reference to strike witches eos",
                    "cowboy to a few friends also added the openings of cowboy bebop and noragami to my anime ost playlist eos",
                    "suggestions would be appreciated i ve been wondering what would make it easier to read some on the text eos",
                    "myanimelistnetanime151re cutie honey short concentrated amp definitely be something that will make alot of people go a bit topsyturvy eos",
                    "this show and 50 holy shit just get some help you arrogant high school bastards gon na be good eos",
                    "sos can this even be considered music https soundcloudcomnottdonlasitkillmebabykillmenobaby funny bc the ed is awesome eos",
                    "of more episodes of development sayonara gundam i ll catch you all in 7 years on green noa 2 eos",
                    "sos if i taught trigonometry every problem would be based on a picture of kaiji eos",
                    "sos evangelion s rebuild eos",
                    "hqgif chihaya gaming http 38mediatumblrcome8bb0f17f565d4d3259bfaf9bf30f853tumblr n7p6ta3ils1rifd3ao1 250gif chihaya drinking https uboachannetotsrc1404236701828png edit the idolm ster spoilers yakusoku https mymixtapemoegzwdxfwebm eos",
                    "was actually fall 2009 when i was about 9 or 10 sword art online la storia della arcana famiglia eos",
                    "think ghost stories was really that popular ps its nowhere in the seas as far as i can tell eos",
                    "sos talk no jutsu eos",
                    "re neither male nor female trap is a whole other gender i m the reason anime was a mistake eos",
                    "sos summer 2014 watched free eternal summer love stage eos",
                    "are you doing it hachi phrasing hachi show the kiss dammit well atleast it happened i ll take it eos",
                    "sos hyperdimensional neptunia is awesome that is all eos",
                    "living with different kinds of monster girls and yet it opened up a lot of things especially on ranime eos",
                    "sos shit i like ryuuji but i got ta vote for best girl eos",
                    "sos archer for president best guy http iimgurcomqi1n3m7gif eos",
                    "sos scryed is amazing it s about as good if not better than gurren laggan eos",
                    "he also participated in the mobile suit gundam char s counterattack spoiler s axis pushback in char s counterattack eos",
                    "sos i literally just read the manga a week ago and man am i hyped right now for it eos",
                    "sos charlotte meets all your conditions plus it meets the bonus eos",
                    "sos never seen this show but that s pretty fuckin sweet eos",
                    "performed automatically please contact the moderators of this subreddit messagecompose to ranime if you have any questions or concerns eos",
                    "sos http myanimelistnetanimelist5hine i enjoy that i can look like a hipster now without doing any work great update eos",
                    "sos wait you re not jordy sublimer flyingbunsofdoom eos",
                    "depth if it was nt for the incredible character that his master was that storyline would be beyond boring eos",
                    "sos white album 2 amagami ss has two defrostings eos",
                    "show it s episodic i guess no one bothered with the recap thus you ca nt find it online eos",
                    "sos erased meets steins gate meets shoujo eos",
                    "commitment to the close up on renge s face for such a prolonged scene was amazing i need crayons eos",
                    "sos no one said gintama yet eos"]

        chatbot_outputs = ["sos my i eos", 
                            "sos the my", 
                            "sos is was would",
                            "sos seems everyone the",
                            "sos that seen holo",
                            "to how 2012 teamsblood yowamushi my seriously   eos",
                            "and guess understand of",
                            "so maniac the all that eos",
                            "sos good",
                            "again can i   is",
                            "ton koutarous   of on",
                            "do think the",
                            "sos s i for",
                            "sos it my to",
                            "sos is who but",
                            "sos it my to",
                            "cowboy go that episode",
                            "suggestions james i to going eos",
                            "myanimelistnetanime151re here amp i think",
                            "and it this plus his eos",
                            "sos this the i eos",
                            "this episodes     all",
                            "sos is my based",
                            "sos s",
                            "Decoded shqgif gaming the i but eos",
                            "Decoded ssos was a or this",
                            "i was a manga wait",
                            "sos a",
                            "sos was moderators to if",
                            "gt ops up   give",
                            "are doing but kiss than eos",
                            "sos neptunia is",
                            "living with show     eos",
                            "killed feel vote i",
                            "sos re of other",
                            "sos is short good for",
                            "the participated from season s",
                            "sos was the week how eos",
                            "seirei the really s i eos",
                            "sos show show be",
                            "performed can moderators messagecompose you",
                            "sos myanimelistnetanimelist5hine is probably a",
                            "sos re of other",
                            "depth of jonny and",
                            "sos album eos",
                            "show mal really bothered i",
                            "sos my shoujo",
                            "sos a of   way",
                            "sos in i"]

        # chatbot_outputs = ["sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos", 
        #                     "sos sos sos"]






        encoded_sentences = self.encoder_decoder_handler.encode_sequence(sentences)
        encoded_replies = self.encoder_decoder_handler.encode_sequence(replies)
        encoded_chatbot_outputs = self.encoder_decoder_handler.encode_sequence(chatbot_outputs)

        print("Encoded sentences: "+str(len(encoded_sentences)))
        print("encoded replies: "+str(len(encoded_replies)))
        print("Encoded chatbot outputs: "+str(len(encoded_chatbot_outputs)))

        bleu_training = countBLEU()
        bleu_training.max_train_len = self.max_train_length

        #gets vectors of each sentence

        for x in range(0, min(len(sentences), len(replies), len(chatbot_outputs))):

            # sentence = sentences[x]
            # reply = replies[x]
            # chatbot_output = chatbot_outputs[x]

            encoded_sentence = encoded_sentences[x]
            encoded_reply = encoded_replies[x]
            encoded_chatbot_output = encoded_chatbot_outputs[x]

            #pads if needed
            for y in range(0, self.max_train_length):
                if y >= len(encoded_sentence):
                    encoded_sentence.append(0)

                if y >= len(encoded_reply):
                    encoded_reply.append(0)

                if y >= len(encoded_chatbot_output):
                    encoded_chatbot_output.append(0)

            # print(encoded_sentence)
            # print(encoded_reply)
            # print(encoded_chatbot_output)
            # print()

            bleu_training.count_BLEU(encoded_chatbot_output, encoded_reply)



        print("Average BLEU score: "+str(float(bleu_training.bleuScore)/len(sentences)))
        print("Average GLEU score: "+str(float(bleu_training.gleuScore)/len(sentences)))









if __name__ == "__main__": 
    print("Loading training, validation, and text data")
    
    chatbot = Chatbot()
    
    num_data_points = 30000
    chatbot.load_data(num_data_points)


    # chatbot.evaluate_sequences()

    # input()


    #if a model hasn't been trained yet, train one
    model_arch_path = "./seq2seq_arch.json"
    model_weight_path = "./seq2seq_weights.h5"
    inference_encoder_arch_path = "./seq2seq_inference_encoder_arch.json"
    inference_encoder_weight_path = "./seq2seq_inference_encoder_weights.h5"
    inference_decoder_arch_path = "./seq2seq_inference_decoder_arch.json"
    inference_decoder_weight_path = "./seq2seq_inference_decoder_weights.h5"


    # if os.path.isfile(model_arch_path)==False:
    
    
    #builds classifier model
    print("Building Encoder and Decoder model")
    training_model, inference_encoder_model, inference_decoder_model = chatbot.build_model()


    print("X_train shape: "+str(chatbot.X_train.shape))
    print("y_train shape: "+str(chatbot.y_train.shape))
    print(chatbot.X_train[0])

    #trains the model
    training_model.fit([chatbot.X_train, chatbot.decoder_X_train], chatbot.decoder_y_train, batch_size=50, epochs=1, shuffle=False)


    # save model
    print("Saving model")
    try:

        ## Saves training model ##
        # serialize model to JSON
        model_json = training_model.to_json()
        with open(model_arch_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        training_model.save_weights(model_weight_path)


        ## Saves inference encoder model ##
        # serialize model to JSON
        inference_encoder = inference_encoder_model.to_json()
        with open(inference_encoder_arch_path, "w") as json_file:
            json_file.write(inference_encoder)
        # serialize weights to HDF5
        inference_encoder_model.save_weights(inference_encoder_weight_path)


        ## Saves inference decoder model ##
        # serialize model to JSON
        inference_decoder = inference_decoder_model.to_json()
        with open(inference_decoder_arch_path, "w") as json_file:
            json_file.write(inference_decoder)
        # serialize weights to HDF5
        inference_decoder_model.save_weights(inference_decoder_weight_path)




        print("Saved model to disk")
    except Exception as error:
        print("Couldn't save model: "+str(error))
    # else:
    #     # training_model = load_model(model_path)

    #     # load json and create model
    #     json_file = open(model_arch_path, 'r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     training_model = model_from_json(loaded_model_json)
    #     # load weights into new model
    #     training_model.load_weights(model_weight_path)
    #     print("Loaded model from disk")

    # evaluate LSTM
    print("Evaluating results")


    bleu_training = countBLEU()
    bleu_training.max_train_len = chatbot.max_train_length

    print("Train data")
    for x in range(0, 50):
        input_sequence = chatbot.X_train[x : x+1]
        predicted_sequence = chatbot.predict_sequence(inference_encoder_model, inference_decoder_model, input_sequence)
        try:
            actual_reply = chatbot.encoder_decoder_handler.decode_sequence(chatbot.y_train[x : x+1][0])
            print("Should have responded with: "+str(actual_reply))
            bleu_training.count_BLEU(predicted_sequence, actual_reply)
        except Exception as error:
            print("Error calculating bleu score: "+str(error))
        print()

    bleu_testing = countBLEU()
    bleu_testing.max_train_len = chatbot.max_train_length

    print("Test data")
    for x in range(0, 50):
        input_sequence = chatbot.X_test[x : x+1]
        predicted_sequence = chatbot.predict_sequence(inference_encoder_model, inference_decoder_model, input_sequence)
        try:
            actual_reply = chatbot.encoder_decoder_handler.decode_sequence(chatbot.y_test[x : x+1][0])
            print("Should have responded with: "+str(actual_reply))
            bleu_testing.count_BLEU(predicted_sequence, actual_reply)
        except Exception as error:
            print("Error calculating bleu score: "+str(error))
        print()



    
    # total, correct = 100, 0
    # bleu = countBLEU(lstm)

    # blue.count_BLEU()
    
    # for i in range(total):
    #     # extract indexes for this batch
    #     X1 = X_test[i]
    #     X2 = test_layer_output[i]
        
    #     y = ku.to_categorical([y_t[i]], num_classes=lstm.vocab_size)
    #     y = np.reshape(y, (y.shape[1], y.shape[2]))       
    #     target = predict_sequence(training_model, X1, X2, lstm.max_train_len, lstm.vocab_size)
        
    #     #print(one_hot_decode(y))
    #     #print(one_hot_decode(target))
    #     interpret(lstm, one_hot_decode(y), one_hot_decode(target))
    #     bleu.count_BLEU(one_hot_decode(y), one_hot_decode(target))
        
    #     if np.array_equal(one_hot_decode(y), one_hot_decode(target)):
    #         correct += 1
    # # print("BLEU score: "+str(bleu.))
    # print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
    
    