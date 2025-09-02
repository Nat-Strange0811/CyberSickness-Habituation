% This get IEC, ERP, ITC and ERSP
%% Made by Gang Li in SJTU 2018-10-16

clc; clear all
eeglab;



%%
dir_in = 'C:\Users\natty\OneDrive\Desktop\Dissertation\Run';
dir_out = 'C:\Users\natty\OneDrive\Documents\Uni - Masters\Dissertation\Data\Pre-Processed EEG Data\Sliced';


filetype='.easy';
%filetype='.set';

if ~exist(dir_out, 'dir')
    mkdir(dir_out);
    fprintf('Created directory %s\n',dir_out)
end
%% Do preproessing

eval(sprintf('this_dir = ''%s/'';', dir_in));
set_files = getAllFiles(this_dir);
Fileidx=strfind(set_files,filetype); Fileidx=find(~cellfun(@isempty,Fileidx)); %Find the cleaned set Files
set_files=set_files(Fileidx);

 %% NAME YOUR OUTPUT HERE

threshold=100;

 for z = 1:length(set_files)
        pathname = sprintf('%s', set_files{z});
        EEG=pop_easy(pathname,1,1,[]);
        %EEG = pop_loadset(pathname);

        ppt_name=extractAfter(pathname,'_');
        ppt_name=extractBefore(ppt_name, '_');
        display(ppt_name);
        %ppt_name=extractAfter(pathname,'Preprocessed\');
        %ppt_name=extractBefore(ppt_name,'.set');
        
        EEG=pop_eegfiltnew(EEG, 0.1, 30); %Basic FIR filter 0.1-30Hz. 
        
        EEG=pop_epoch(EEG,{'1'},[60 120]);
        
        EEG=pop_rmbase(EEG,[],[]); % baseline corrected to 
        
      
        %EEG=pop_eegthresh(EEG, 1, [1:EEG.nbchan], -threshold, threshold, -250, -0.002, 0, 1); % Remove the trias that exceed threshold -100 to 100 uV
        %EEG=pop_select(EEG, 'nochannel' , {'EXT'});
        EEG=pop_chanedit(EEG,'load','C:\Users\natty\OneDrive\Documents\MATLAB\eeglab_current\eeglab2025.0.0\plugins\neuroelectrics\Locations\cybersickness study.ced');
        EEG=pop_chanedit(EEG,'changefield',{6 'theta' -25 'radius' 0.6});
        %EEG=pop_runica(EEG,'runica'); %Run ICA
        %outfile=sprintf('%s',ppt_name);
        pop_saveset(EEG, ppt_name, dir_out);

        clear ppt_name;
        clear EEG;

 end