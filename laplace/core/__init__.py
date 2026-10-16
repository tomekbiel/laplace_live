"""
Modules copied unchanged from the research repository.

Nothing in this package may be edited. A change here silently moves the live
strategy away from the model that was trained and measured, and the drift
would not show up in any test. Improvements belong in the modules beside this
package, which may wrap these functions but never alter them.

Verify with:

    diff -r laplace/core <research-repo>/core
"""
