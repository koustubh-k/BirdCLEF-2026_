if not _single_solution:
    submission = f_add()
    submission.to_csv(f"submission.csv", index = True)
    submission