model = Net().cuda()

fold_pred = []
for path in ['model_576_0.pt', 'model_576_1.pt', 'model_576_2.pt', 'model_576_3.pt', 'model_576_4.pt']:
    model.load_state_dict(torch.load(path))
    model.eval()
    pred = []
    with torch.no_grad():
        for data in test_loader:
            data = data[0].cuda()
            scores = model(data)
            pred.append(scores.data.cpu().numpy())
    
    pred = np.vstack(pred)
    fold_pred.append(pred)
    
    
    
fold_pred = np.mean(fold_pred, axis=0)
test_df['label'] = fold_pred.argmax(1) + 1
test_df['label'] = test_df['label'].apply(lambda x: 'd' + str(x))
test_df['image'] = test_df['path'].apply(lambda x: x.split('/')[-1])
test_df['image'] = test_df['image'].apply(lambda x: x.split('.')[0]+'.jpg')
test_df[['image','label']].to_csv('submit.csv', index=None)
