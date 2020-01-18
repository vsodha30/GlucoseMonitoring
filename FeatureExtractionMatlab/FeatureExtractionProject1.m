clc
clear all
load('WorkspaceforFeatureExtractionProject1')
% yyaxis left
cgmDatenum1 = table2array(CGMDatenumLunchPat1);
cgmSeries1 = table2array(CGMSeriesLunchPat1);
insulinBasal1 = table2array(InsulinBasalLunchPat1);
insulinBolus1 = table2array(InsulinBolusLunchPat1);
insulinDatenum1 = table2array(InsulinDatenumLunchPat1);
plot(cgmDatenum1(20,:), cgmSeries1(20,:));
hold on
stem(insulinDatenum1(20,:), insulinBasal1(20,:) + insulinBolus1(20,:), 'bo');

featureMatrix1 = [];
featureMatrix2 = [];
featureMatrix3 = [];
% for row = 1:size(cgmSeries1,1)-1
for row = 1:size(insulinBolus1,1)-1
%     [G,H] = max(cgmSeries1(row+1, :));
    
    [G,H] = max(insulinBolus1(row+1, :));
%     timeMax = cgmDatenum1(row, (H(1)));
    timeMax = insulinDatenum1(row, (H(1)));
%     counter = size;
    while(isnan(G(1)) | G(1)== 'NaN' | isnan(timeMax) | timeMax == 'NaN' )
        insulinBolus1(row+1, H(1)) = -1;        
        insulinDatenum1(row+1, H(1)) = -1;
        [G,H] = max(insulinBolus1(row+1, :));
        timeMax = insulinDatenum1(row+1, H(1));
    end
    [G,H] = max(insulinBolus1(row+1, :));
    timeMax = insulinDatenum1(row, (H(1)));
    row;
%     mySize = size(cgmDatenum1(row, :));
%     colSize = size(cgmSeries1,2);     
    colSize = size(insulinBolus1,2);     
    while(colSize >= 1)
%         if(isnan(cgmDatenum1(row, colSize)))
        if(isnan(insulinDatenum1(row, colSize)) | insulinDatenum1(row, colSize) == 'NaN')
            colSize = colSize - 1;
        else
            break;
        end 
        
    end 
%     colSize
%     cgmDatenum1(row, colSize)
    insulinDatenum1(row, colSize)
%     datetime(timeMax - cgmDatenum1(row, colSize), 'ConvertFrom', 'datenum');
%     featureVector1 = [timeMax - cgmDatenum1(row, 1)];
    d1 = datestr(datetime(timeMax, 'ConvertFrom', 'datenum'));
    d2 = datestr(datetime(insulinDatenum1(row, colSize), 'ConvertFrom', 'datenum'));
    strcat(d1, {',      '}, d2)
%     cgmDatenum1(row, 1)
%     diffDateTime = datetime(timeMax - cgmDatenum1(row, colSize), 'ConvertFrom', 'datenum')
    diffDateTime = datetime(timeMax - insulinDatenum1(row, colSize), 'ConvertFrom', 'datenum')
%     tempTime = second(diffDateTime)
    temphour = hms(diffDateTime)
    tempminute = minute(diffDateTime)
%     tempTime =  ()* (24 * 60 * 60)
    featureVector1 = [temphour*60 + tempminute]
%     featureVector1 = [datetime(timeMax - cgmDatenum1(row, 1), 'ConvertFrom', 'datenum')];
    featureVector2 = [G(1)];
    featureVector3 = [H(1)];
    featureMatrix1 = [featureMatrix1; featureVector1];
    featureMatrix2 = [featureMatrix2; featureVector2];
    featureMatrix3 = [featureMatrix3; featureVector3];
    
    (H(1));
    
%     datetime(737085, 'ConvertFrom', 'datenum')
    
    
%     cgmVel = cgmSeries1(row, 1:end-1) - cgmSeries1(row, 2:end);
%     [G,H] = find(cgmVel == 0);
%     zeroCrossing = G(1);
    
end
myVar = var(featureMatrix1);
myMean = mean(featureMatrix1);

histogram(featureMatrix1);
pd = fitdist(featureMatrix1,'Normal')
x = 7:8; 
p = pdf(pd,x);

xlim([0 200])
plot(x, p)
finalFeatureMatrix = []
for rowNum = 1:size(featureMatrix1,1)
    if featureMatrix1(rowNum) >= myMean - 30 & featureMatrix1(rowNum) <= myMean + 30
        isOnTime = 1;
    else 
        isOnTime = 0;
    end
    
    featureVector1 = [featureMatrix1(rowNum,1) myMean , 30 , isOnTime];
    finalFeatureMatrix = [finalFeatureMatrix; featureVector1];
end
% finalFeatureMatrix = [finalFeatureMatrix, mean(finalFeatureMatrix(:,4))]
