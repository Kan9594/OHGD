function [hat_y_t] = run_OHGD(Y, X, id_list)
%%
%Code for Online Harmonizing Gradient Descent
%The definition of GD_P/GD_N in code differs slightly from that in the original work, 
%but the concept of harmonizing the gradient norm remains consistent.
%--------------------------------------------------
%INPUT: Y---1*N lables (-1, +1)
%            X---T*d data matrix
%            id_list---the order of data in streams
%OUTPUT: hat_y_t---the prediction
%--------------------------------------------------

%% Initilization
w = zeros(1,size(X,2));
Weight = 1;
GD_P = 0;GD_N = 0;
Num_P = 0;Num_N = 0;

%% Online learning
for t = 1:length(id_list)
    ID  = id_list(t);
    x_t = X(ID,:);
    y_t = Y(ID);

%% Predicting
    f_t = w*x_t';
    if (f_t>=0)
        hat_y_t(t) = 1;
    else
        hat_y_t(t) = -1;
    end
    
%% Updating
    eta_t   = 1/sqrt(t);
    l_t = max(0,1-y_t*f_t);
    if y_t==-1
        Num_N = Num_N + 1;
    else
        Num_P = Num_P + 1;
    end

    if (l_t > 0)
        if t>2
            if y_t ==1
                Weight = l_t*Num_N/max(Num_P,1)*2*GD_N/max(GD_P+GD_N,1);
                if Weight > 2e2
                    Weight = 2e2;
                elseif Weight < 1e-2
                    Weight = 1e-2;
                end
            else
                Weight = l_t*2*GD_P/max(GD_P+GD_N,1);
                if Weight > 2e2
                    Weight = 2e2;
                elseif Weight < 1e-2
                    Weight = 1e-2;
                end
            end
        end
        w = w + Weight*eta_t*y_t*x_t;
        
        % GI Calculating
        if (l_t > 0)
            if y_t == 1
                GD_P = GD_P + norm(Weight*eta_t*y_t*x_t);
            else
                GD_N = GD_N + norm(Weight*eta_t*y_t*x_t);
            end
        end
    end
end

%THE END
