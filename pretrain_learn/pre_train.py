
import torch


#1. 遍历训练轮次
#遍历批次
# 重置损失梯度
#计算损失
#反向传播 计算 损失梯度
#根据损失梯度更新模型权重
#打印
#可视化

def train_model_simple(model , train_loader , val_loader , optimizer ,device ,num_epochs ,
                       eval_freq , eval_iter , start_context , tokenizer):
    train_losses , val_losses , track_token_seen = [] , [] , []
    tokens_seen , global_step = 0 , -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader:
            optimizer.zero_grad() #重置损失梯度
            loss = cal_loss_batch(
                input_batch , target_batch , model ,device
            )
            loss.backward()
            optimizer.setp()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0 :
                train_loss , val_loss = evaluate_model(
                    model , train_loader , val_loader , device , eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d} )"
                      f"Train loss {train_loss:.3f}"
                      f"val loss {val_loss:.3f}"
                      )

                test



