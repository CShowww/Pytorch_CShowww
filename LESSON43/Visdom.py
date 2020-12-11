'''
lines:single trace
from visdom import Visdom
viz = Visdom()
#第一个参数Y，第二个参数X
viz.line([0.],[0.],win = 'train_loss',opts = dict(title = 'train loss'))
viz.line([loss.item()],[global_step],win = 'train_loss',update = 'append')

#lines:multi-traces
viz = Visdom()
viz.line([[0.0,0.0]],[0.],win = 'test',opts = dict(title = 'test loss & acc',
                                   legend = ['loss','acc']))
viz.line([[test_loss,correct / len(test_loader.dataset)]],
         [global_step],win = 'test',update = 'append')


#visualX
viz = Visdom()
viz.images(data.view(-1,1,28,28),win = 'x')
viz.text(str(pred.detach().cpu().numpy()),win = 'pred',opts = dict(title = 'pred'))
'''


