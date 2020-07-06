import torch

def train_model(model, train_input, train_target, train_classes, criterion, optimizer, mini_batch_size, nb_epochs):
    """
    Trains a model using the given train data sets and hyperparameters.
    """
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output_main, output_aux1, output_aux2 = model(train_input.narrow(0, b, mini_batch_size))
            if (output_aux1 is None and output_aux2 is None):
                loss = criterion(output_main, train_target.narrow(0, b, mini_batch_size))
            elif (output_aux1 is not None and output_aux2 is not None):
                loss_main = criterion(output_main, train_target.narrow(0, b, mini_batch_size))
                loss_aux1 = criterion(output_aux1, train_classes[:, 0].narrow(0, b, mini_batch_size))
                loss_aux2 = criterion(output_aux2, train_classes[:, 1].narrow(0, b, mini_batch_size))
                loss = loss_main + loss_aux1 + loss_aux2
            else:
                sys.exit("One of the two auxiliary losses is None while the other is not")
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, test_input, test_target, test_classes, mini_batch_size):
    """
    Computes the number of errors with the provided test data sets.
    """
    nb_target_errors = 0
    nb_digit_errors = 0
    nb_classes_errors = 0
    classes_bool = False
    for b in range(0, test_input.size(0), mini_batch_size):
        output_main, output_aux1, output_aux2 = model(test_input.narrow(0, b, mini_batch_size))
        if output_aux1 is None and output_aux2 is None:
            highest_numbers_indices_main = output_main.max(1)[1]
            for i in range(highest_numbers_indices_main.size(0)):
                if highest_numbers_indices_main[i] != test_target[b + i]:
                    nb_target_errors += 1
        elif output_aux1 is not None and output_aux2 is not None:
            if not classes_bool:
                classes_bool = True
            highest_numbers_indices_main = output_main.max(1)[1]
            highest_numbers_indices_aux1 = output_aux1.max(1)[1]
            highest_numbers_indices_aux2 = output_aux2.max(1)[1]
            for i in range(highest_numbers_indices_main.size(0)):
                if highest_numbers_indices_main[i] != test_target[b + i]:
                    nb_target_errors += 1
                first_digit_err = False
                second_digit_err = False
                if highest_numbers_indices_aux1[i] != test_classes[b + i][0]:
                    nb_digit_errors += 1
                    first_digit_err = True
                if highest_numbers_indices_aux2[i] != test_classes[b + i][1]:
                    nb_digit_errors += 1
                    second_digit_err = True
                if first_digit_err or second_digit_err:
                    nb_classes_errors += 1
        else:
            sys.exit("One of the two auxiliary losses is None while the other is not")
    if classes_bool:
        return nb_target_errors, nb_digit_errors, nb_classes_errors
    else:
        return nb_target_errors, None, None

def get_stats(model, optimizer, criterion, mini_batch_size, nb_epochs,
              train_input, train_target, train_classes, test_input, test_target, test_classes):
    """
    Returns the test error rate after having trained a model with the training sets and computed the number of errors using the testing data sets.
    """
    print('Training ' + model.__class__.__name__ + '...')

    train_model(model, train_input, train_target, train_classes, criterion, optimizer, mini_batch_size, nb_epochs)

    nb_target_errors, nb_digit_errors, nb_classes_errors = compute_nb_errors(
        model, test_input, test_target, test_classes, mini_batch_size)
    
    target_error_rate = nb_target_errors / test_target.size(0) * 100
    digit_error_rate = -1
    classes_error_rate = -1
    if nb_digit_errors is not None and nb_classes_errors is not None:
        digit_error_rate = (nb_digit_errors / (test_classes.size(0) * test_classes.size(1))) * 100
        classes_error_rate = nb_classes_errors / test_classes.size(0) * 100

    if digit_error_rate == -1 and classes_error_rate == -1:
        print('Target error rate (differences with the labels of the test_target set) is {:d} out of {:d} ({:.02f}%)'.format(
            nb_target_errors, test_target.size(0), target_error_rate))
    else:
        print('Target error rate (differences with the labels of the test_target set) is {:d} out of {:d} ({:.02f}%)'.format(
            nb_target_errors, test_target.size(0), target_error_rate))
        print('Classes error rate (differences with the labels of the test_classes set) is {:d} out of {:d} ({:.02f}%) with {:d} out of {:d} single digit errors ({:.02f}%)'.format(
            nb_classes_errors, test_classes.size(0), classes_error_rate, 
            nb_digit_errors, test_classes.size(0) * test_classes.size(1), digit_error_rate))
        
    return target_error_rate
