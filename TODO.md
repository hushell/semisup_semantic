x 1. check if x_drop affects (it seems x_drop = 0 the best, x_drop0.99 leads to high L1 and disturbs CE)
x 2. baseline without L1 loss
x 3. consistency loss + self-sup (consist_loss cannot distinguish a partial vs correct)
4. contrastive loss
5. play with coeff and lr for x_drop = 0.99 (coeff higher)
