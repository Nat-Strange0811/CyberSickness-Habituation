clc; clear all;
eeglab;
%*IMPORTANT* BE sure that your electrodes are in this EXACT order, or you are comparing mush to mush
elecs = {'FZ','PZ','P3','P4','CZ','CP5','CP6'};
elecs = upper(elecs);
 
% indicate which electrodes to include in your frontal,central and back cluster
fronts1 = {'FZ'};
centrals1 = {'CZ'};
backs1={'PZ'};
backs2={'P3'};
backs3={'P4'};
backs4={'CP5'};
backs5={'CP6'};

frontNames = {'Frontal'};
centralNames = {'Central'};
backNames = {'Central_Posterior','Left_PIVC_Posterior','Right_PIVC_Posterior','Left_TPJ_Posterior','Right_TPJ_Posterior'};

for t=1:length(frontNames) % for example, 1:3 if you had 3 different frontal electrode clusters
    clear selectedElecs;
    eval(['fronts = fronts', num2str(t), ';'])
    for i = 1:length(fronts)
        for ii = 1:length(elecs)
            if strcmp(fronts{i},elecs{ii});
                selectedElecs(1,i) = ii;
            end
        end
    end
    eval(['fronts',num2str(t),' = selectedElecs;']);
end
for t=1:length(centralNames) % for example, 1:3 if you had 3 different frontal electrode clusters
    clear selectedElecs;
    eval(['centrals = centrals', num2str(t), ';'])
    for i = 1:length(centrals)
        for ii = 1:length(elecs)
            if strcmp(centrals{i},elecs{ii});
                selectedElecs(1,i) = ii;
            end
        end
    end
    eval(['centrals',num2str(t),' = selectedElecs;']);
end
for t=1:length(backNames) % for example, 1:3 if you had 3 different posterior electrode clusters 
    clear selectedElecs;
    eval(['backs = backs', num2str(t), ';'])
    for i = 1:length(backs)
        for ii = 1:length(elecs)
            if strcmp(backs{i},elecs{ii});
                selectedElecs(1,i) = ii;
            end
        end
    end
    eval(['backs',num2str(t),' = selectedElecs;']);
end

low_center_frequency1 = [1 3]; 
low_center_frequency2 = [3 5]; 
low_center_frequency3 = [5 7];
low_center_frequency4 = [7 9];
low_center_frequency5 = [9 11]; 
low_center_frequency6 = [11 13]; 
low_center_frequency7 = [13 15];
low_center_frequency8 = [15 17];
low_center_frequency9 = [17 19];
low_center_frequency10 = [19 21];

freq_name = {'delta2','theta4','theta6','alpha8','alpha10','alpha12','beta14','beta16','beta18','beta20'};
anglefactor = 1; % need to be unit quantity


Run_ITC=true;
Run_IEC=true;
Run_RBP=true;

dir_in = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\Pre-Processed EEG Data\Sliced\';
IEC_dir_out ='C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\IEC Data\Sliced\';
ITC_dir_out = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\ITC Data\Sliced\';
RBP_dir_out = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\RBP Data\Sliced\';
%dir_out ='F:\Glasgow University\tDCS Motion Sickness\Data\Chair Study\EEG data\tACS\Active\Useful data\LiveDemo4Nat\IEC\';

filetype='.set';

set_files = getAllFiles(dir_in);
Fileidx=strfind(set_files,filetype); Fileidx=find(~cellfun(@isempty,Fileidx)); %Find the cleaned set Files
set_files=set_files(Fileidx);

row_start=1;
%row_end=0;
%avg_row_start=1;


for z = 1:length(set_files)
    pathname = sprintf('%s', set_files{z});
    EEG = pop_loadset(pathname);
    ppt_name=extractAfter(pathname,'Sliced\');
    ppt_name=extractBefore(ppt_name,'.');
    
    outfiledir_rbp=sprintf('%s%s',RBP_dir_out,ppt_name);
    outfiledir_itc=sprintf('%s%s',ITC_dir_out,ppt_name);
    outfiledir_iec=sprintf('%s%s',IEC_dir_out,ppt_name);

    result=zeros(29,length(EEG.trials));
    indx=0;
    
    %Select the epoch
    for trialNum=1:EEG.trials
        OUTEEG=pop_select(EEG,'trial',trialNum);
        OUTEEG.xmin=0;
        OUTEEG.xmax=OUTEEG.pnts/OUTEEG.srate;
        OUTEEG=eeg_checkset(OUTEEG);
        OUTEEG=eeg_regepochs(OUTEEG,'recurrence',2,'limits',[0 2]);
        row_id=1;
        
        if Run_ITC==true
            tlimits=[OUTEEG.times(1), OUTEEG.times(end)];
            ERSPs=struct('data',[],'itc',[],'powbase',[],'times',[],'freqs',[],'itcphase',[]);
            ERSPs(size(OUTEEG.data,1)).data=[];

            for elec=1:size(OUTEEG.data,1)
                for ind=1:OUTEEG.trials
                    new_data(elec,(OUTEEG.pnts*ind-OUTEEG.pnts+1):(OUTEEG.pnts*ind))=OUTEEG.data(elec,1:OUTEEG.pnts,ind);
                end 
                [ERSPs(elec).data,ERSPs(elec).itc,ERSPs(elec).powbase,ERSPs(elec).times,ERSPs(elec).freqs,~,~,ERSPs(elec).itcphase]= ...
                    timef(new_data(elec,:),OUTEEG.pnts,tlimits,OUTEEG.srate,0,'timesout',500,'maxfreq',30,'baseline',0,'plotersp','off','itctype','coher','plotitc','off','freqscale','linear');
            end

            for el=1:length(ERSPs)
                ERSP.power(el,:,:)=ERSPs(el).data;
                ERSP.powbase(el,:,:)=ERSPs(el).powbase;
                ERSP.itc(el,:,:)=ERSPs(el).itc;
                ERSP.channels={OUTEEG.chanlocs.labels};
                ERSP.times=ERSPs(1).times;
                ERSP.freqs=ERSPs(1).freqs;
            end
           
            deltaIdx = find(ERSP.freqs>1 & ERSP.freqs<3);  % delta=2
            theta_1_Idx = find(ERSP.freqs>3 & ERSP.freqs<5);  % theta=4
            theta_2_Idx = find(ERSP.freqs>5 & ERSP.freqs<7);  % theta=6
            alpha_1_Idx = find(ERSP.freqs>7 & ERSP.freqs<9);  % alpha=8
            alpha_2_Idx = find(ERSP.freqs>9 & ERSP.freqs<11);  % alpha=10
            alpha_3_Idx = find(ERSP.freqs>11 & ERSP.freqs<13);  % alpha=12
            beta_1_Idx = find(ERSP.freqs>13 & ERSP.freqs<15);  % low beta=14
            beta_2_Idx = find(ERSP.freqs>15 & ERSP.freqs<17);  % low beta=16
            beta_3_Idx = find(ERSP.freqs>17 & ERSP.freqs<19);  % low beta=18
            beta_4_Idx = find(ERSP.freqs>19 & ERSP.freqs<21);  % low beta=20
            for n=1:OUTEEG.nbchan  
                A=mean(squeeze(ERSP.itc(n,deltaIdx,1:length(ERSP.times))));
                B=mean(squeeze(ERSP.itc(n,theta_1_Idx,1:length(ERSP.times))));
                C=mean(squeeze(ERSP.itc(n,theta_2_Idx,1:length(ERSP.times))));
                D=mean(squeeze(ERSP.itc(n,alpha_1_Idx,1:length(ERSP.times))));
                E=mean(squeeze(ERSP.itc(n,alpha_2_Idx,1:length(ERSP.times))));
                F=mean(squeeze(ERSP.itc(n,alpha_3_Idx,1:length(ERSP.times))));
                G=mean(squeeze(ERSP.itc(n,beta_1_Idx,1:length(ERSP.times))));
                H=mean(squeeze(ERSP.itc(n,beta_2_Idx,1:length(ERSP.times))));
                I=mean(squeeze(ERSP.itc(n,beta_3_Idx,1:length(ERSP.times))));
                J=mean(squeeze(ERSP.itc(n,beta_4_Idx,1:length(ERSP.times))));
                
                output_itc(row_id,trialNum)=mean(A);
                output_itc(row_id+1,trialNum)=mean(B);
                output_itc(row_id+2,trialNum)=mean(C);
                output_itc(row_id+3,trialNum)=mean(D);
                output_itc(row_id+4,trialNum)=mean(E);
                output_itc(row_id+5,trialNum)=mean(F);
                output_itc(row_id+6,trialNum)=mean(G);
                output_itc(row_id+7,trialNum)=mean(H);
                output_itc(row_id+8,trialNum)=mean(I);
                output_itc(row_id+9,trialNum)=mean(J);
                row_id=row_id+10;
                clear A B C D E F G H I J
            end
        end

        if Run_IEC==true
            row_id=1;
            for fq=1:length(freq_name)
                for ff = 1:length(frontNames) % for example, 1:3 if you had 3 different frontal electrodes
                    for bb = 1:length(backNames) % for example, 1:3 if you had 3 different posterior electrodes
                        eval(['chans1 = fronts' num2str(ff) ';']);
                        eval(['chans2 = backs' num2str(bb) ';']);
                        connecName = [frontNames{ff} '_' backNames{bb}];
                        %% NAME YOUR OUTPUT HERE
                        % Define data
                        data1 = double(OUTEEG.data(chans1, :, :));
                        data2 = double(OUTEEG.data(chans2, :, :));
                        sampling_rate = OUTEEG.srate;

                        % Initialize
                        full_plv = zeros(size(data1, 3), size(data1, 2));

                        % Loop across events
                        for eventIt = 1:size(data1, 3)
                            % Get channel one data
                            signal = [zeros(1, 5000) mean(squeeze(data1(:, :, eventIt)), 1) zeros(1, 5000)];
                            locutoff1=eval(sprintf('low_center_frequency%d(1)',fq));
                            hicutoff1=eval(sprintf('low_center_frequency%d(2)',fq));
                            temp = eegfilt(signal, sampling_rate, eval(sprintf('low_center_frequency%d(1)',fq)), eval(sprintf('low_center_frequency%d(2)',fq)));
                            %temp = eegfilt(temp, sampling_rate, [], eval(sprintf('low_center_frequency%d(2)',fq)));
                            chan_one_phase = angle(hilbert(temp)).*anglefactor;  %times 2 for theta to alpha -- change this factor
                            chan_one_phase = chan_one_phase(5001:(end-5000));
                            clear temp signal

                            % Get channel two data
                            signal = [zeros(1, 5000) mean(squeeze(data2(:, :, eventIt)), 1) zeros(1, 5000)];
                            locutoff2=eval(sprintf('low_center_frequency%d(1)',fq));
                            hicutoff2=eval(sprintf('low_center_frequency%d(2)',fq));
                            temp = eegfilt(signal, sampling_rate, eval(sprintf('low_center_frequency%d(1)',fq)), eval(sprintf('low_center_frequency%d(2)',fq)));   %set to low center frequency2 to compute cross freq coherence, set to center frequency1 to compute coherence within same freq band
                            %temp = eegfilt(temp, sampling_rate, [], eval(sprintf('low_center_frequency%d(2)',fq)));
                            chan_two_phase = angle(hilbert(temp));
                            chan_two_phase = chan_two_phase(5001:(end-5000));
                            clear temp signal

                            % Get phase difference between channels
                            full_plv(eventIt, :) = exp(1i * (chan_one_phase - chan_two_phase));

                            clear chan_*
                        end
                        %clear eventIt sampling_rate chans low_center_frequency data
                        clear eventIt sampling_rate


                        % Initialize
                        intertrial_plv = zeros(1, size(full_plv, 2));

                        % Loop across time
                        for timeIt = 1:size(full_plv, 2);
                            intertrial_plv(timeIt) = abs(sum(full_plv(:, timeIt), 'double'))...
                                ./ size(full_plv, 1);
                        end

                        clear timeIt
                        output_iec(row_id,trialNum)=mean(intertrial_plv);

                        clear intertrial_plv

                     
                        row_id=row_id+1;
                       
                        clear output

                    end
                end
            end
        end

        if Run_RBP == true
            fprintf('%d\n', indx)
            indx=indx+1;
            fs=EEG.srate;L=1024;

            chan_mean_deltaPower = zeros(EEG.nbchan, 1);
            chan_mean_thetaPower = zeros(EEG.nbchan, 1);
            chan_mean_alphaPower = zeros(EEG.nbchan,1);
            chan_mean_betaPower = zeros(EEG.nbchan, 1);

            x=EEG.data(1,:,indx);
            [s,f,t]=stft(x,fs,'Window',hann(1000,'periodic'),'OverlapLength',500,'FFTLength',L);

            deltaPower = zeros(EEG.nbchan, length(t));
            thetaPower = zeros(EEG.nbchan, length(t));
            alphaPower = zeros(EEG.nbchan, length(t));
            betaPower = zeros(EEG.nbchan, length(t));

            chan_rbp=zeros(EEG.nbchan,4);

            s=0;f=0;t=0;

            for n=1:EEG.nbchan 
                x=EEG.data(n,:,indx);
                [s,f,t]=stft(x,fs,'Window',hann(1000,'periodic'),'OverlapLength',500,'FFTLength',L);
                P2 = abs(s/L);
                P1 = P2(L/2:L,:);
                P1(2:end-1) = 2*P1(2:end-1);
                freq=f(L/2:L);


                deltaIdx = find(freq>=0.5 & freq<=3);  % delta=0.5-3
                thetaIdx = find(freq>=4 & freq<=7);  % theta=4-7
                alphaIdx = find(freq>=8 & freq<=12);  % alpha=8-13
                betaIdx = find(freq>=13 & freq<=20);  % low beta=13-18

                A=P1(deltaIdx,1:length(t));
                if(length(deltaIdx)==1)
                    A=A';
                    deltaPower(n,:)=A;
                else
                    deltaPower(n,:)=mean(A);
                end

                B=P1(thetaIdx,1:length(t));
                if(length(thetaIdx)==1)
                    B=B';
                    thetaPower(n,:)=B;
                else
                    thetaPower(n,:)=mean(B);
                end

                C=P1(alphaIdx,1:length(t));
                if(length(alphaIdx)==1)
                    C=C';
                    alphaPower(n,:)=C;
                else
                    alphaPower(n,:)=mean(C);
                end

                D=P1(betaIdx,1:length(t));
                if(length(betaIdx)==1)
                    D=D';
                    betaPower(n,:)=D;
                else
                    betaPower(n,:)=mean(D);
                end

                chan_mean_deltaPower(n,1)=mean(deltaPower(n,:),2);
                chan_mean_thetaPower(n,1)=mean(thetaPower(n,:),2);
                chan_mean_alphaPower(n,1)=mean(alphaPower(n,:),2);
                chan_mean_betaPower(n,1)=mean(betaPower(n,:),2);

                A=0;B=0;C=0;D=0;P1=0;P2=0;
            end

            grand_average_delta=mean(chan_mean_deltaPower);
            grand_average_theta=mean(chan_mean_thetaPower);
            grand_average_alpha=mean(chan_mean_alphaPower);
            grand_average_beta=mean(chan_mean_betaPower);

            for i=1:EEG.nbchan
                chan_rbp_delta=100*chan_mean_deltaPower(i)/sum(chan_mean_deltaPower(i)+chan_mean_thetaPower(i)+chan_mean_alphaPower(i)+chan_mean_betaPower(i));
                chan_rbp_theta=100*chan_mean_thetaPower(i)/sum(chan_mean_deltaPower(i)+chan_mean_thetaPower(i)+chan_mean_alphaPower(i)+chan_mean_betaPower(i));
                chan_rbp_alpha=100*chan_mean_alphaPower(i)/sum(chan_mean_deltaPower(i)+chan_mean_thetaPower(i)+chan_mean_alphaPower(i)+chan_mean_betaPower(i));
                chan_rbp_beta=100*chan_mean_betaPower(i)/sum(chan_mean_deltaPower(i)+chan_mean_thetaPower(i)+chan_mean_alphaPower(i)+chan_mean_betaPower(i));
                chan_rbp(i,:)=[chan_rbp_delta,chan_rbp_theta,chan_rbp_alpha,chan_rbp_beta];
            end

            result(1,indx)=chan_rbp(1,1); %Fz-delta
            result(2,indx)=chan_rbp(1,2); %Fz-theta
            result(3,indx)=chan_rbp(1,3); %Fz-alpha;
            result(4,indx)=chan_rbp(1,4); %Fz-beta

            result(5,indx)=chan_rbp(2,1); %Cz-delta
            result(6,indx)=chan_rbp(2,2); %Cz-theta
            result(7,indx)=chan_rbp(2,3); %Cz-alpha;
            result(8,indx)=chan_rbp(2,4); %Cz-beta
            result(9,indx)=chan_rbp(2,4)/chan_rbp(2,2); %Cz-beta/theta;

            result(10,indx)=chan_rbp(3,1); %Pz-delta
            result(11,indx)=chan_rbp(3,2); %Pz-theta
            result(12,indx)=chan_rbp(3,3); %Pz-alpha;
            result(13,indx)=chan_rbp(3,4); %Pz-beta

            result(14,indx)=chan_rbp(4,1); %P3-delta
            result(15,indx)=chan_rbp(4,2); %P3-theta
            result(16,indx)=chan_rbp(4,3); %P3-alpha;
            result(17,indx)=chan_rbp(4,4); %P3-beta

            result(18,indx)=chan_rbp(5,1); %Cp5-delta
            result(19,indx)=chan_rbp(5,2); %Cp5-theta
            result(20,indx)=chan_rbp(5,3); %Cp5-alpha;
            result(21,indx)=chan_rbp(5,4); %Cp5-beta

            result(22,indx)=chan_rbp(6,1); %P4-delta
            result(23,indx)=chan_rbp(6,2); %P4-theta
            result(24,indx)=chan_rbp(6,3); %P4-alpha;
            result(25,indx)=chan_rbp(6,4); %P4-beta

            result(26,indx)=chan_rbp(7,1); %Cp6-delta
            result(27,indx)=chan_rbp(7,2); %Cp6-theta
            result(28,indx)=chan_rbp(7,3); %Cp6-alpha;
            result(29,indx)=chan_rbp(7,4); %Cp6-beta
        end
    end
    output_rbp = result;
    if Run_RBP == true
        save(outfiledir_rbp,'output_rbp')
    end
    if Run_ITC == true
        save(outfiledir_itc,'output_itc');
    end
    if Run_IEC == true
        save(outfiledir_iec,'output_iec');
    end
    clear output_itc output_iec output_rbp
end
