x 1. check if x_drop affects (it seems x_drop = 0 the best, x_drop0.99 leads to high L1 and disturbs CE)
x 2. baseline without L1 loss
x 3. consistency loss + self-sup (consist_loss cannot distinguish a partial vs correct)
4. contrastive loss (predict which forward is from clean image)
5. play with coeff and lr for x_drop = 0.99 (coeff higher)
6. other types of selfsup (rotation, other noises, style transfer)
7. read Lilian Weng's survey
8. think why pretext/selfsup work? (rot, mask make representation aware of foreground)
