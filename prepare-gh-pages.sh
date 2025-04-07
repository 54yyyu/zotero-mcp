#!/bin/bash

# Move docs-site files to root directory for GitHub Pages
echo "Moving documentation files to root directory..."

# Copy all files from docs-site to root
cp -r docs-site/* .
cp docs-site/.nojekyll .

# Remove the docs-site directory to avoid duplication
rm -rf docs-site

echo "Files moved successfully!"
