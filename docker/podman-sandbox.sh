#!/bin/sh
# Confine rootless Podman to one directory, so everything this project creates on a shared
# host (images, containers, layers) can be listed and removed in one step.
#
#   docker/podman-sandbox.sh setup [DIR]     # default DIR=$HOME/geogym; writes DIR/pm wrapper
#   docker/podman-sandbox.sh status [DIR]
#   docker/podman-sandbox.sh clean [DIR]     # dry run: shows what would be removed
#   docker/podman-sandbox.sh clean [DIR] --yes
#
# Then use DIR/pm exactly like `podman` (DIR/pm build ..., DIR/pm run ...). No root, no system
# packages, no services, no changes outside DIR (Podman's runtime state lives in /run/user/$UID).
set -eu
cmd=${1:-status}
dir=${2:-$HOME/geogym}

case "$cmd" in
setup)
    mkdir -p "$dir/containers/storage"
    cat > "$dir/storage.conf" <<EOF
[storage]
driver = "overlay"
graphroot = "$dir/containers/storage"
runroot = "/run/user/$(id -u)/geogym-containers"
EOF
    cat > "$dir/pm" <<EOF
#!/bin/sh
export CONTAINERS_STORAGE_CONF="$dir/storage.conf"
exec podman "\$@"
EOF
    chmod +x "$dir/pm"
    echo "ready: use $dir/pm in place of podman"
    ;;
status)
    "$dir/pm" ps -a --format '{{.Names}}  {{.Status}}'
    "$dir/pm" images --format '{{.Repository}}:{{.Tag}}  {{.Size}}'
    du -sh "$dir"
    ;;
clean)
    echo "would remove: all containers and images in $dir/containers, then $dir itself"
    "$dir/pm" ps -a --format '  container {{.Names}}' || true
    "$dir/pm" images --format '  image {{.Repository}}:{{.Tag}}  {{.Size}}' || true
    du -sh "$dir" || true
    if [ "${3:-}" = "--yes" ]; then
        "$dir/pm" rm -af >/dev/null
        "$dir/pm" rmi -af >/dev/null
        # Layer files are owned by sub-UIDs; delete them inside the user namespace.
        "$dir/pm" unshare rm -rf "$dir/containers"
        rm -rf "$dir"
        echo "removed $dir"
    else
        echo "dry run; add --yes to remove"
    fi
    ;;
*)
    echo "usage: $0 setup|status|clean [DIR] [--yes]" >&2
    exit 2
    ;;
esac
