for sigName = {'d', 'e_train', 'e_test', 'e_t'}
    sigName = sigName{1};
    sigNames = {sigName, [sigName,'_lpf']};
    for newSigName = sigNames
        newSigName = newSigName{1};
        cellObj = [newSigName,'_PrePost'];
        matrObj = [cellObj,'_ch'];
    
        disp([matrObj,'(trIdx,:,1) = ',newSigName,'(tr + ((-nBeforeTrig):0));'])
        disp([matrObj,'(trIdx,:,2) = ',newSigName,'(tr + (  0:nBeforeTrig ));'])

%        disp([cellObj,'{idx} = ',matrObj,';'])
    end
end