function model = varma_fx(yTrain, maxLag)

dimInput = size(yTrain, 2);
varmModel = varm(dimInput, maxLag);
estModel = estimate(varmModel, yTrain);

model = estModel;

end