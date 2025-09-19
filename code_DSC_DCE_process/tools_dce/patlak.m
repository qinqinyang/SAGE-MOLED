     %%%%%%%%%%%%% extented Tofts model %%%%%%%%%
     % Cp --- blood plasma contrast agent concentration.
     % s1 --- DCE time courses
     % R1b0, R1i ---  blood and tissue R1
     % Output: pes-- blood fraction, EES, and Ktrans
     %         sfit -- fitting curve;  
     
     function [pes, sfit,pescv] = patlak(Cp,s1, R1i, FA, r1, TR, ft)



         % Fitting parameter [vp, ktrans]
        
          beta0 = [0.02,   0.01];
          lb =    [0.001,  1e-5];
          ub =    [0.24,   1 ]; %%wfy
          %ub =    [1,   1 ];

%           options =optimset('Display','off','TolFun',max(s1)*1e-8...
%                                             ,'MaxFunEvals',200);

          s1_0 = mean(s1(1:4)); 
          %[pes,~, residual,~,~,~,J] = lsqnonlin(@(p) DCE_S_gen_Patlak(s1_0,Cp, R1i, p, FA, r1, TR, ft)-s1,beta0,lb,ub,options);
          [pes,~, residual,~,~,~,J] = lsqnonlin(@(p) DCE_S_gen_Patlak(s1_0,Cp, R1i, p, FA, r1, TR, ft)-s1,beta0,lb);
          pescv = nlparci(pes, residual,'Jacobian',J);
          sfit = s1 + residual;
      
     end