'''
#momentum:注意Adam优化器自带
optimizer = torch.optim.SGD(model.parameters().args.lr,
                               momentum = args.momentum,
                               weight_decay = args.weight_decay)
scheduler = ReduceLOonPlateau(optimizer,'min')
for epoch in xrange(args.start_epoch,args.epochs):
    train(train_loader,model,criterion,optimizer,epoch)
    result_avg,loss_val = validate(val_loader,model,criterion,epoch)
    scheduler.step(loss_val)
'''

'''
#lr_decay：先选大一点的lr,然后逐步减小
#方法一：观测loss，如果loss平缓一段时间，则减少
optimizer = torch.optim.SGD(model.parameters().args.lr,
                               momentum = args.momentum,
                               weight_decay = args.weight_decay)
scheduler = ReduceLOonPlateau(optimizer,'min')
for epoch in xrange(args.start_epoch,args.epochs):
    train(train_loader,model,criterion,optimizer,epoch)
    result_avg,loss_val = validate(val_loader,model,criterion,epoch)
    scheduler.step(loss_val)
    
    
    
#方法二：每多少个epoch,衰减多少，简单直接，gamma为原来的倍数
scheduler = StepLR(optimizer,step_size = 30,gamma = 0.1)
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
'''